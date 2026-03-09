from preprocessing.image_preprocessing import convert_image_to_tensor
from preprocessing.pil_preprocessing import crop_pil, letterbox_resize
from preprocessing.tensor_preprocessing import heatmaps_to_keypoints, normalize_keypoints_xy


class InferencePipeline:
    def __init__(self, bbox_detection, keypoint_detection, pose_detection, keypoint_image_size=(256, 256)):
        self.bbox_detection = bbox_detection
        self.keypoint_detection = keypoint_detection
        self.pose_detection = pose_detection
        self.keypoint_image_size = keypoint_image_size

    def run_pipe(self, pil_image):
        # image to tensor for bounding box detection (stage 1)
        tensor = convert_image_to_tensor(pil_image)
        bbox = self.bbox_detection(tensor.unsqueeze(0)) 

        # crop and resize for keypoint detection (stage 2)
        cropped_img = crop_pil(pil_image, bbox)
        resized_img = letterbox_resize(cropped_img, size=self.keypoint_image_size)
        keypoint_tensor = convert_image_to_tensor(resized_img)

        heatmaps = self.keypoint_detection(keypoint_tensor.unsqueeze(0))
        # convert heatmaps to keypoints
        keypoints = heatmaps_to_keypoints(heatmaps)

        # normalize to [0,1] (must match training)
        H = heatmaps.size(2)
        W = heatmaps.size(3)
        keypoints = normalize_keypoints_xy(keypoints, H, W)

        # pose classification, inputting matrix of keypoints of shape (18, 3) (stage 3)
        pose = self.pose_detection(keypoints)

        return pose