python train/end_to_end_train.py \
    --root_dir "datasets/orvile/tennis-player-actions-dataset/versions/1/Tennis Player Actions Dataset for Human Pose Estimation" \
    --bbox_checkpoint "exports/bbox_best.pt" \
    --keypoint_checkpoint "exports/keypoint_best_state_dict.pt" \
    --pose_checkpoint "exports/pose_best_predicted.pt" \
    --batch_size 16 \
    --epochs 40 \
    --unfreeze_stage2_epoch 4 \
    --unfreeze_stage1_epoch 8
