# FastAPI Python Backend for Model Inference

We use FastAPI here to link communication between frontend demo and backend model inference. The only endpoint defined here is the POST `/predict` endpoint where:
- Request body: Raw image to be used for inference.
- Response body: Model predictions in the format: 
    - Stage One
        - bbox_x: number
        - bbox_y: number
        - bbox_width: number
        - bbox_height: number
        - cropped_bbox_image: image (Cropped image according to bbox boundaries)

    - Stage Two
        - keypoint_positions: number[][] ([[Keypoint 1 x, Keypoint 1 y, Keypoint 1 visibility], [Keypoint 2 x, Keypoint 2 y, Keypoint 2 visibility], ..., [Keypoint 18 x, Keypoint 18 y, Keypoint 18 visibility]])
        - keypoint_aggregated_heatmap_url: string (Image of all keypoint heatmaps displayed against black background.)
        - keypoint_cropped_image: image (Image of cropped image with keypoints overlayed)

    - Stage Three
        - backhand_conf: number
        - forehand_conf: numbe
        - ready_position_conf: number
        - serve_conf: number
        - class_prediction_bar_graph: image (Bar graph of class prediction confidences)

    - Final Prediction
        - overlayed_image: image (Overlayed image with bounding box, keypoints, and final class prediction)
    - These predictions map directly to `tpd-demo`'s Prediction model. 

All the images and graphs to be returned here has all already been implemented in `notebooks/training`. To find the corresponding logic for each images, they each belong in their own stage's notebook. Copy that logic to here to use for generation of images.

