from preprocessing.pil_preprocessing import crop_pil, letterbox_resize, norm_bbox_to_xyxy_pixels
from preprocessing.tensor_preprocessing import heatmaps_to_keypoints, normalize_keypoints_xy
from torchvision import transforms


class InferencePipeline:
    def __init__(
        self,
        bbox_detection,
        keypoint_detection,
        pose_detection,
        bbox_image_size=(256, 256),
        keypoint_image_size=(256, 256),
    ):
        self.bbox_detection = bbox_detection
        self.keypoint_detection = keypoint_detection
        self.pose_detection = pose_detection
        self.bbox_image_size = bbox_image_size
        self.keypoint_image_size = keypoint_image_size

    def run_pipe(self, pil_image):
        bbox_device = next(self.bbox_detection.parameters()).device
        keypoint_device = next(self.keypoint_detection.parameters()).device
        pose_device = next(self.pose_detection.parameters()).device

        # image to tensor for bounding box detection (stage 1)
        bbox_transform = transforms.Compose([
            transforms.Resize(self.bbox_image_size),
            transforms.ToTensor(),
        ])
        tensor = bbox_transform(pil_image.copy()).to(bbox_device)
        bbox = self.bbox_detection(tensor.unsqueeze(0)).squeeze(0)
        image_width, image_height = pil_image.size
        bbox_xyxy = norm_bbox_to_xyxy_pixels(bbox, image_width, image_height)

        # crop and resize for keypoint detection (stage 2)
        cropped_img = crop_pil(pil_image, bbox_xyxy)
        resized_img = letterbox_resize(cropped_img, size=self.keypoint_image_size)
        keypoint_tensor = convert_image_to_tensor(resized_img).to(keypoint_device)

        heatmaps = self.keypoint_detection(keypoint_tensor.unsqueeze(0))
        # convert heatmaps to keypoints
        keypoints = heatmaps_to_keypoints(heatmaps)

        # normalize to [0,1] (must match training)
        H = heatmaps.size(2)
        W = heatmaps.size(3)
        keypoints = normalize_keypoints_xy(keypoints, H, W)

        # pose classification, inputting matrix of keypoints of shape (18, 3) (stage 3)
        pose = self.pose_detection(keypoints.to(pose_device))

        return pose