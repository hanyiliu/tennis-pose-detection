from preprocessing.image_preprocessing import convert_image_to_tensor
from preprocessing.pil_preprocessing import crop_pil, letterbox_resize

class InferencePipeline:
    def __init__(self, bbox_detection, keypoint_detection, pose_detection, keypoint_image_size=(256, 256)):
        self.bbox_detection = bbox_detection
        self.keypoint_detection = keypoint_detection
        self.pose_detection = pose_detection
        self.keypoint_image_size = keypoint_image_size

    def run_pipe(self, pil_image):
        # image to tensor for bounding box detection
        tensor = convert_image_to_tensor(pil_image)
        bbox = self.bbox_detection(tensor.unsqueeze(0)) 

        # crop and resize for keypoint detection
        cropped_img = crop_pil(pil_image, bbox)
        resized_img = letterbox_resize(cropped_img, size=self.keypoint_image_size)
        keypoint_tensor = convert_image_to_tensor(resized_img)
        keypoints = self.keypoint_detection(keypoint_tensor.unsqueeze(0))

        # pose classification inputting matrix of keypoints of shape (18, 3)
        pose = self.pose_detection(keypoints)

        return pose