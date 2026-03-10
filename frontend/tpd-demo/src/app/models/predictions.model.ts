export interface Predictions {
    // Stage One
    bbox_x: number;
    bbox_y: number;
    bbox_width: number;
    bbox_height: number;
    cropped_bbox_image_url: string; // URL to cropped image according to bbox boundaries.

    // Stage Two
    keypoint_positions: number[][]; // [[Keypoint 1 x, Keypoint 1 y, Keypoint 1 visibility], [Keypoint 2 x, Keypoint 2 y, Keypoint 2 visibility], ..., Keypoint 18 x, Keypoint 18 y, Keypoint 18 visibility]]
    keypoint_aggregated_heatmap_url: string; // URL to image of all keypoint heatmaps displayed against black background.
    keypoint_cropped_image_url: string; // URL to image of cropped image with keypoints overlayed

    // Stage Three
    backhand_conf: number;
    forehand_conf: number;
    ready_position_conf: number;
    serve_conf: number;
    class_prediction_bar_graph_url: string; // URL to bar graph of class prediction confidences

}