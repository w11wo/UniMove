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

for city in istanbul kuwait_city moscow new_york petaling_jaya sao_paulo; do
    python main.py \
        --device cuda:0 \
        --city $city \
        --target_city $city \
        --train_root traj_dataset/massive_steps/train \
        --val_root traj_dataset/massive_steps/val \
        --test_root traj_dataset/massive_steps/test \
        --B 4
done

# for city in  jakarta; do
# done