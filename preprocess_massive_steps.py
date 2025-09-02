from datetime import datetime, timedelta
from argparse import ArgumentParser
from pathlib import Path
import csv

from sklearn.preprocessing import StandardScaler, normalize
from tqdm import tqdm
import pandas as pd
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--city",
        type=str,
        required=True,
        choices=[
            "beijing",
            "istanbul",
            "jakarta",
            "kuwait_city",
            "melbourne",
            "moscow",
            "new_york",
            "petaling_jaya",
            "sao_paulo",
            "shanghai",
            "sydney",
            "tokyo",
        ],
    )
    parser.add_argument("--massive_steps_path", type=str, default="../Massive-STEPS/")
    return parser.parse_args()


def preprocess_trajectories(df, output_dir, city, split):
    """Preprocess trajectories from Massive-STEPS CSV to UniMove plaintext format."""

    def _group_trajectories(user_df, time_window=timedelta(hours=72)):
        """Group check-ins into trajectories based on time window.
        UniMove uses 72 hours as the default time window.
        Also collapse timestamps into day of week and 30-minute intervals.
        """
        user_df = user_df.sort_values("timestamp").copy()

        traj_ids, time_intervals, days_of_week = [], [], []
        curr_traj_id = 0
        traj_start = None

        for ts in user_df["timestamp"]:
            ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            if traj_start is None:
                traj_start = ts

            # check if still inside trajectory window
            if ts >= traj_start + time_window:
                # start a new trajectory
                curr_traj_id += 1
                traj_start = ts

            traj_ids.append(f"{user_df['user_id'].iloc[0]}_{curr_traj_id}")

            # collapse to nearest day of week
            days_of_week.append(ts.weekday())
            # collapse to nearest 30-minute interval
            minutes_since_midnight = ts.hour * 60 + ts.minute
            interval_idx = minutes_since_midnight // 30  # 0..47
            time_intervals.append(interval_idx)

        user_df["traj_id"] = traj_ids
        user_df["day_of_week"] = days_of_week
        user_df["time_interval"] = time_intervals
        return user_df

    def _convert_to_plaintext(traj_df):
        """Convert a trajectory dataframe to plaintext format, following the UniMove format:
        `venue_id,day_of_week,time_interval;venue_id,day_of_week,time_interval;...`
        """
        assert traj_df["day_of_week"].min() >= 0 and traj_df["day_of_week"].max() <= 6
        assert traj_df["time_interval"].min() >= 0 and traj_df["time_interval"].max() <= 47

        trajs = []
        for venue_id, day_of_week, time_interval in zip(
            traj_df["venue_id"], traj_df["day_of_week"], traj_df["time_interval"]
        ):
            token = f"{venue_id},{day_of_week},{time_interval}"
            trajs.append(token)
        return ";".join(trajs)

    output_path = output_dir / split / f"{city}_{split}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    user_df = df.copy().groupby("user_id").apply(_group_trajectories, include_groups=True).reset_index(drop=True)
    trajectories = user_df.groupby("traj_id").apply(_convert_to_plaintext, include_groups=False)
    trajectory_user_ids = [traj_id.split("_")[0] for traj_id in trajectories.index]

    with open(output_path, "w") as f:
        for user_id, trajectory in zip(trajectory_user_ids, trajectories):
            f.write(f"{user_id} {trajectory}\n")


def preprocess_locations(all_df, all_categories):
    """Preprocess location features from Massive-STEPS CSV to UniMove format.
    Location features include:
    - City-level category counts (937-dim)
    - City-level category distributions (937-dim)
    - Normalized latitude and longitude (2-dim)
    - Visit frequency rank (1-dim, 0-7)
    Total: 1877-dim feature vector for each POI.

    Returns a numpy array of shape [num_poi, 1877].
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

    # fill missing lat/lon with city lat/lon
    all_df["latitude"] = all_df["latitude"].fillna(all_df["venue_city_latitude"])
    all_df["longitude"] = all_df["longitude"].fillna(all_df["venue_city_longitude"])

    # get venue ranks based on visit frequency
    venue2rank = _get_venue2rank(all_df["venue_id"])
    all_df["rank"] = all_df["venue_id"].apply(lambda x: venue2rank[x])

    poi_df = (
        all_df[["venue_id", "venue_category_id", "latitude", "longitude", "venue_city", "rank"]]
        .drop_duplicates(subset=["venue_id"])
        .reset_index(drop=True)
    )
    num_poi = max(poi_df["venue_id"]) + 1

    # normalize lat/lon as normal dist
    scaler = StandardScaler()
    poi_df[["latitude", "longitude"]] = scaler.fit_transform(poi_df[["latitude", "longitude"]])

    # get city-level category counts and distributions
    counts = poi_df.groupby(["venue_city", "venue_category_id"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=all_categories, fill_value=0)
    normalized_counts = normalize(counts, norm="l1")

    city2cats = counts.to_dict(orient="index")
    city2cats = {city: list(counts.loc[city]) for city in counts.index}
    city2cat_dists = {city: list(normalized_counts[i]) for i, city in enumerate(counts.index)}

    all_poi_info = []
    for venue_id in tqdm(range(num_poi), desc="Getting POI info"):
        if venue_id not in poi_df["venue_id"].values:
            raise ValueError(f"Venue ID {venue_id} not found in POI dataframe.")

        poi_row = poi_df[poi_df["venue_id"] == venue_id]

        poi_venue_city = poi_row["venue_city"].values[0]
        poi_cat_count = city2cats[poi_venue_city]
        poi_cat_dist = city2cat_dists[poi_venue_city]
        # array of cat_count + cat_dist + lat + lon + rank
        poi_info = np.array(
            poi_cat_count + poi_cat_dist + poi_row[["latitude", "longitude", "rank"]].values.flatten().tolist()
        )
        assert poi_info.shape[0] == len(all_categories) * 2 + 3  # [1877], where 1877 = 937*2 + 2 + 1
        all_poi_info.append(poi_info)

    all_poi_info = np.stack(all_poi_info)
    assert all_poi_info.shape == (num_poi, len(all_categories) * 2 + 3)  # [N, 1877]

    return all_poi_info


def main(args):
    location_dir = Path("location_feature")
    trajectory_dir = Path("traj_dataset") / "massive_steps"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    massive_steps_path = Path(args.massive_steps_path)
    data_path = massive_steps_path / "data" / args.city

    train_df = pd.read_csv(data_path / f"{args.city}_checkins_train.csv")
    test_df = pd.read_csv(data_path / f"{args.city}_checkins_test.csv")
    validation_df = pd.read_csv(data_path / f"{args.city}_checkins_validation.csv")

    # remap venue_id to be continuous integers starting from 0, ignoring unseen POIs in training sets
    all_df = pd.concat([train_df, test_df, validation_df], ignore_index=True)
    venue2poi = {venue: idx for idx, venue in enumerate(all_df["venue_id"].unique())}
    train_df["venue_id"] = train_df["venue_id"].apply(lambda x: venue2poi[x])
    test_df["venue_id"] = test_df["venue_id"].apply(lambda x: venue2poi[x])
    validation_df["venue_id"] = validation_df["venue_id"].apply(lambda x: venue2poi[x])
    all_df["venue_id"] = all_df["venue_id"].apply(lambda x: venue2poi[x])

    # create trajectories and save to plaintext files
    preprocess_trajectories(train_df, trajectory_dir, args.city, "train")
    preprocess_trajectories(test_df, trajectory_dir, args.city, "test")
    preprocess_trajectories(validation_df, trajectory_dir, args.city, "val")

    # load all POI categories
    all_categories = []
    with open(massive_steps_path / "semantic-trails" / "categories.csv", "r") as f:
        lines = csv.reader(f, delimiter=",")
        for line in lines:
            category, *_ = line
            all_categories.append(category)

    # create location_feature array
    all_poi_info = preprocess_locations(all_df, all_categories)
    np.save(location_dir / f"vocab_{args.city}.npy", all_poi_info)


if __name__ == "__main__":
    args = parse_args()
    main(args)
