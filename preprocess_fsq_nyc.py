from argparse import ArgumentParser
from pathlib import Path
import csv

from sklearn.preprocessing import StandardScaler, normalize
from tqdm import tqdm
import pandas as pd
import numpy as np
import pyproj


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../LLM4POI/datasets/nyc/raw/")
    parser.add_argument("--semantic_trails_path", type=str, default="../semantic-trails/")
    return parser.parse_args()


def preprocess_trajectories(df, output_dir, split):
    """Preprocess trajectories from FSQ-NYC CSV to UniMove plaintext format."""

    def _convert_to_plaintext(traj_df):
        """Convert a trajectory dataframe to plaintext format, following the UniMove format:
        `poi_id,day_of_week,time_interval;poi_id,day_of_week,time_interval;...`
        """
        assert traj_df["day_of_week"].min() >= 0 and traj_df["day_of_week"].max() <= 6
        assert traj_df["time_interval"].min() >= 0 and traj_df["time_interval"].max() <= 47

        traj_df = traj_df.sort_values(by="local_time")

        trajs = []
        for poi_id, day_of_week, time_interval in zip(
            traj_df["POI_id"], traj_df["day_of_week"], traj_df["time_interval"]
        ):
            token = f"{poi_id},{day_of_week},{time_interval}"
            trajs.append(token)
        return ";".join(trajs)

    output_path = output_dir / split / f"nyc_{split}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # collapse timestamp to day of week and 30-min interval
    df["day_of_week"] = pd.to_datetime(df["local_time"]).dt.weekday
    df["time_interval"] = (
        pd.to_datetime(df["local_time"]).dt.hour * 60 + pd.to_datetime(df["local_time"]).dt.minute
    ) // 30

    trajectories = df.groupby("trajectory_id").apply(_convert_to_plaintext, include_groups=False)

    with open(output_path, "w") as f:
        for trail_id, trajectory in zip(trajectories.index, trajectories):
            f.write(f"{trail_id} {trajectory}\n")


def preprocess_locations(all_df, all_categories):
    """Preprocess location features from FSQ-NYC CSV to UniMove format.
    Location features include:
    - City-level category counts (162-dim)
    - City-level category distributions (162-dim)
    - Normalized latitude and longitude (2-dim)
    - Visit frequency rank (1-dim, 0-7)
    Total: 324-dim feature vector for each POI.

    Returns a numpy array of shape [num_poi, 324].
    """

    def _get_rank(p):
        """Convert visit frequency percentile to rank 0-7. Follows Table 2 in the UniMove paper."""
        if p < 0.01:
            return 0
        elif p < 0.05:
            return 1
        elif p < 0.10:
            return 2
        elif p < 0.20:
            return 3
        elif p < 0.40:
            return 4
        elif p < 0.60:
            return 5
        elif p < 0.80:
            return 6
        else:
            return 7

    def _get_venue2rank(venue_visits):
        """Get venue_id to rank mapping based on visit frequency."""
        counts = pd.Series(venue_visits).value_counts()
        counts = counts.sort_values(ascending=False)
        freqs = counts / counts.sum()
        cumulative = freqs.cumsum()
        ranks = cumulative.apply(_get_rank)
        venue2rank = {venue: rank for venue, rank in zip(counts.index, ranks)}

        return venue2rank

    # get venue ranks based on visit frequency
    venue2rank = _get_venue2rank(all_df["POI_id"])
    all_df["rank"] = all_df["POI_id"].apply(lambda x: venue2rank[x])

    # project lat/lon to metres using UTM
    proj = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)  # WGS84  # Web Mercator ~ meters
    all_df["x_m"], all_df["y_m"] = proj.transform(all_df["longitude"].values, all_df["latitude"].values)

    # assign each POI to a grid of 500x500 metres
    cell_size_m = 500
    all_df["grid_x"] = (all_df["x_m"] // cell_size_m).astype(int)
    all_df["grid_y"] = (all_df["y_m"] // cell_size_m).astype(int)
    all_df["area_cluster"] = all_df["grid_x"].astype(str) + "_" + all_df["grid_y"].astype(str)

    poi_df = (
        all_df[["POI_id", "POI_catname", "latitude", "longitude", "area_cluster", "rank"]]
        .drop_duplicates(subset=["POI_id"])
        .reset_index(drop=True)
    )
    num_poi = max(poi_df["POI_id"]) + 1

    # normalize lat/lon as normal dist
    scaler = StandardScaler()
    poi_df[["latitude", "longitude"]] = scaler.fit_transform(poi_df[["latitude", "longitude"]])

    # get area-level category counts and distributions
    counts = poi_df.groupby(["area_cluster", "POI_catname"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=all_categories, fill_value=0)
    normalized_counts = normalize(counts, norm="l1")

    city2cats = counts.to_dict(orient="index")
    city2cats = {city: list(counts.loc[city]) for city in counts.index}
    city2cat_dists = {city: list(normalized_counts[i]) for i, city in enumerate(counts.index)}

    all_poi_info = []
    for poi_id in tqdm(range(num_poi), desc="Getting POI info"):
        if poi_id not in poi_df["POI_id"].values:
            raise ValueError(f"POI ID {poi_id} not found in POI dataframe.")

        poi_row = poi_df[poi_df["POI_id"] == poi_id]

        poi_venue_city = poi_row["area_cluster"].values[0]
        poi_cat_count = city2cats[poi_venue_city]
        poi_cat_dist = city2cat_dists[poi_venue_city]
        # array of cat_count + cat_dist + lat + lon + rank
        poi_info = np.array(
            poi_cat_count + poi_cat_dist + poi_row[["latitude", "longitude", "rank"]].values.flatten().tolist()
        )
        assert poi_info.shape[0] == len(all_categories) * 2 + 3
        all_poi_info.append(poi_info)

    all_poi_info = np.stack(all_poi_info)
    assert all_poi_info.shape == (num_poi, len(all_categories) * 2 + 3)

    return all_poi_info


def main(args):
    location_dir = Path("location_feature")
    trajectory_dir = Path("traj_dataset") / "fsq"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    semantic_trails_path = Path(args.semantic_trails_path)
    data_path = Path(args.data_path)

    train_df = pd.read_csv(data_path / "NYC_train.csv")
    test_df = pd.read_csv(data_path / "NYC_test.csv")
    validation_df = pd.read_csv(data_path / "NYC_val.csv")

    # remap venue_id to be continuous integers starting from 0, ignoring unseen POIs in training sets
    all_df = pd.concat([train_df, test_df, validation_df], ignore_index=True)
    venue2poi = {venue: idx for idx, venue in enumerate(all_df["POI_id"].unique())}
    train_df["POI_id"] = train_df["POI_id"].apply(lambda x: venue2poi[x])
    test_df["POI_id"] = test_df["POI_id"].apply(lambda x: venue2poi[x])
    validation_df["POI_id"] = validation_df["POI_id"].apply(lambda x: venue2poi[x])
    all_df["POI_id"] = all_df["POI_id"].apply(lambda x: venue2poi[x])

    # create trajectories and save to plaintext files
    preprocess_trajectories(train_df, trajectory_dir, "train")
    preprocess_trajectories(test_df, trajectory_dir, "test")
    preprocess_trajectories(validation_df, trajectory_dir, "val")

    # load all POI categories
    all_categories = set()
    category2schema = {}
    with open(semantic_trails_path / "mapping.csv", "r") as f:
        lines = csv.reader(f, delimiter=",")
        for line in lines:
            category_id, _, schema = line
            schema = schema.replace("schema:", "")
            category2schema[category_id] = schema
            all_categories.add(schema)

    # manually add missing categories
    category2schema["4e51a0c0bd41d3446defbb2e"] = "CivicStructure"  # Ferry
    # remap POI_catid to schema.org's 162 categories
    all_df["POI_catid"] = all_df["POI_catid"].apply(lambda x: category2schema[x])

    # create location_feature array
    all_poi_info = preprocess_locations(all_df, all_categories)
    np.save(location_dir / "vocab_nyc.npy", all_poi_info)


if __name__ == "__main__":
    args = parse_args()
    main(args)
