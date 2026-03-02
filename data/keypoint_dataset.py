import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import torch

class TennisKeypointDataset(Dataset):
    """A Pytorch Dataset mapping cropped images to keypoints

    """    
    def __init__(
        self,
        root_dir: str,
        annotation_dir: str = "annotations",
        image_transform: Optional[Callable[[Image.Image], Any]] = transforms.ToTensor(),
        keypoint_transform: Optional[Callable[..., Any]] = None
    ):
        """Set up dataset to hold correct data for extraction

        Args:
            root_dir (str): Path to the root `dataset` folder
            annotation_dir (str): Name of the subdir that all annotation json files are in
            image_transform (transforms, optional): Any transforms to be applied to the cropped images (such as letterbox resizing and convert PIL to tensor). Defaults to only transforms.ToTensor().
            keypoint_transform (transforms, optional): Any transforms to be applied to the keypoint matrix (such as convert to heatmap). Defaults to None.
        """        
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.keypoint_transform = keypoint_transform

        self.data: List[Dict] = []
        annotation_path = os.path.join(root_dir, annotation_dir)
        for name in sorted(os.listdir(annotation_path)):
            path = os.path.join(annotation_path, name)
            if not os.path.isfile(path):
                continue
        
            with Path(path).open("r", encoding="utf-8") as f:
                annotation_data = json.load(f)
            
            id_to_img = {img["id"]: img for img in annotation_data.get("images", [])}
            
            for ann in annotation_data.get("annotations", []):
                image_id = ann["image_id"]
                img_info = id_to_img.get(image_id)
                if img_info is None:
                    continue
                
                # COCO bbox format: [x_min, y_min, width, height]
                bbox = ann.get("bbox", None)
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                _, _, w, h = bbox
                if w <= 0 or h <= 0:
                    continue
                
                # JSON path: "../images/backhand/B_001.jpeg"
                # we want to get the relative path to root_dir
                rel_path = img_info.get("path", img_info["file_name"])
                rel_path = rel_path.replace("../", "")
                img_path = os.path.join(root_dir, rel_path)

                # Get keypoints
                keypoints = ann.get("keypoints", None)
                if not isinstance(keypoints, list) or len(keypoints) != 54:
                    continue
                
                self.data.append(
                    {
                        "img_path": img_path,
                        "bbox": bbox,
                        "keypoints": keypoints,   
                    }
                )

    def __len__(self):
        """Return size of the dataset

        Returns:
            int: How many samples are in the dataset
        """        
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get data point at corresponding index. 

        Args:
            idx (_type_): _description_

        Returns:
            cropped_img: The cropped image of the tennis player after all transforms
            keypoints (torch.tensor): A tensor of shape (18,3) where each row is one keypoint
        """        
        img_path = self.data[idx]["img_path"]
        x, y, w, h = self.data[idx]["bbox"]
        keypoints_list = self.data[idx]["keypoints"]

        # Process cropped image
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        left = max(0, int(round(x)))
        top = max(0, int(round(y)))
        right = min(W, int(round(x + w)))
        bottom = min(H, int(round(y + h)))
        cropped_img = image.crop((left, top, right, bottom))

        if self.image_transform is not None:
            cropped_img = self.image_transform(cropped_img)

        # Process normalized keypoints
        keypoints = np.array(keypoints_list, dtype=np.float32).reshape(18, 3) # turns flat list of 54 into (18, 3), where 18 is number of keypoints and 3 is (x,y,visibility) for each keypoint.
        width = max(float(w), 1.0)
        height = max(float(h), 1.0)
        keypoints[:, 0] = (keypoints[:, 0] - x) / width  # normalize x to bbox-local coords
        keypoints[:, 1] = (keypoints[:, 1] - y) / height # normalize y to bbox-local coords

        if self.keypoint_transform is not None:
            keypoints = self.keypoint_transform(keypoints, float(w), float(h))
            
        return cropped_img, torch.tensor(keypoints, dtype=torch.float32)