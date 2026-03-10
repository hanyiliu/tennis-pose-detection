python train/pose_train_predicted.py \
     --feature_path saved_models/stage3_predicted_attention_keypoints.pt \
     --epochs 100 \
     --batch_size 64 \
     --train_split 0.7 \
     --val_split 0.15 \
     --test_split 0.15 \
     --seed 42 \
     --early_stop_patience 12 \
     --lr 0.0005 \
     --weight_decay 0.001 \
     --hidden_dim 384 \
     --dropout 0.4 \
     --visibility_threshold 0.0 \
     --label_smoothing 0.1 \
     --xy_noise_std 0.001 \
     --joint_dropout 0.0 \
     