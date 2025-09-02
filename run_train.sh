for city in beijing melbourne shanghai sydney tokyo; do
    python main.py \
        --device cuda:0 \
        --city $city \
        --target_city $city \
        --train_root traj_dataset/massive_steps/train \
        --val_root traj_dataset/massive_steps/val \
        --test_root traj_dataset/massive_steps/test \
        --B 8
done

for city in kuwait_city moscow; do
    python main.py \
        --device cuda:0 \
        --city $city \
        --target_city $city \
        --train_root traj_dataset/massive_steps/train \
        --val_root traj_dataset/massive_steps/val \
        --test_root traj_dataset/massive_steps/test \
        --B 4
done

# for city in istanbul jakarta new_york petaling_jaya sao_paulo; do
# done