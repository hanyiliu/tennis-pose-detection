python train/build_pose_stage2_features.py \
    --stage2_checkpoint "exports/keypoint_attention_best_state_dict.pt" \
    --stage2_model_type attention \
    --save_path "saved_models/stage3_predicted_attention_keypoints.pt"
