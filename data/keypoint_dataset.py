import json
import os
from typing import List, Dict
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import torch

class TennisKeypointDataset(Dataset):
    """A Pytorch Dataset mapping cropped images to keypoints

    """    
    def __init__(self, root_dir: str, annotation_dir: str = "annotations", transform=transforms.ToTensor()):
        """Set up dataset to hold correct data for extraction

        Args:
            root_dir (str): Path to the root `dataset` folder
            annotation_dir (str): Name of the subdir that all annotation json files are in
            transform (_type_, optional): Any transforms to be applied to the cropped images (such as letterbox resizing and convert PIL to tensor). Defaults to only transforms.ToTensor().
        """        
        self.root_dir = root_dir
        self.transform = transform

        self.data: List[Dict] = []
        annotation_path = os.path.join(root_dir, annotation_dir)
        for name in sorted(os.listdir(annotation_path)):
            path = os.path.join(annotation_path, name)
            if not os.path.isfile(path):
                continue
        
            with open(path, "r") as f:
                annotation_data = json.load(f)
            
            id_to_img = {img["id"]: img for img in annotation_data["images"]}
            
            for ann in annotation_data["annotations"]:
                image_id = ann["image_id"]
                img_info = id_to_img[image_id]
                
                # COCO bbox format: [x_min, y_min, width, height]
                bbox = ann["bbox"]
                
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

        if self.transform is not None:
            cropped_img = self.transform(cropped_img)

        # Process normalized keypoints
        keypoints = np.array(keypoints_list, dtype=np.float32).reshape(18,3) # turns flat list of 54 into (18, 3), where 18 is number of keypoints and 3 is (x,y,visibility) for each keypoint.
        width = max(float(w), 1.0)
        height = max(float(h), 1.0)
        keypoints[:, 0] /= width # normalize x
        keypoints[:, 1] /= height # normalize y

        return cropped_img, torch.tensor(keypoints, dtype=torch.float32)