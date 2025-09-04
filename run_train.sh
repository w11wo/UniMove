for city in beijing istanbul jakarta kuwait_city melbourne moscow new_york petaling_jaya sao_paulo shanghai sydney tokyo; do
    python main.py \
        --device cuda:0 \
        --city $city \
        --target_city $city \
        --train_root traj_dataset/massive_steps/train \
        --val_root traj_dataset/massive_steps/val \
        --test_root traj_dataset/massive_steps/test \
        --B 4
done