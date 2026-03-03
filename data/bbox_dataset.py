from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torch
import json
import os
from typing import List, Dict

import argparse
from dataclasses import dataclass
import torch.nn as nn
from torchvision import transforms
from models.bbox_detection import BBoxDetectionModel

# PyTorch requires __init__, __len__, and __getitem__ for dataset classes
class TennisBBoxDataset(Dataset):
    """
    loads tennis images and bounding boxes from COCO JSON annotations
    returns
        image: FloatTensor of shape (3, H, W)
        bbox: FloatTensor of shape (4,) in [x_min, y_min, w, h] format
    """
    def __init__(self, root_dir: str, annotation_files: List[str], transform=None):
        
        self.root_dir = root_dir
        # image preprocessing
        self.transform = transform
        # samples example:
        # [ {"img_path": "img1.jpg", width: 1280, height: 720, "bbox": [x, y, w, h]}, {"img_path": "img2.jpg",  width: 1280, height: 720, "bbox": [x, y, w, h]}]
        self.samples: List[Dict] = []
        
        for ann_path in annotation_files:
            full_ann_path = os.path.join(root_dir, ann_path)
            with open(full_ann_path, "r") as f:
                data = json.load(f)
                
            id_to_img = {img["id"]: img for img in data["images"]}
            
            for ann in data["annotations"]:
                image_id = ann["image_id"]
                img_info = id_to_img[image_id]
                
                # COCO bbox format: [x_min, y_min, width, height]
                bbox = ann["bbox"]
                
                # JSON path: "../images/backhand/B_001.jpeg"
                # we want to get the relative path to root_dir
                rel_path = img_info.get("path", img_info["file_name"])
                rel_path = rel_path.replace("../", "")
                img_path = os.path.join(root_dir, rel_path)
                
                self.samples.append(
                    {
                        "img_path": img_path,
                        "width": img_info["width"],
                        "height": img_info["height"],
                        "bbox": bbox,    
                    }
                )
        
    # returns how many samples are in the dataset
    # for batching
    def __len__(self):
        return len(self.samples)
    
    # image, bbox = dataset[i]
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # loads image from disk and ensures there are always 3 channels (RGB)
        img = Image.open(sample["img_path"]).convert("RGB")
        
        W, H = img.size
        x, y, w, h = sample["bbox"]
        
        # normalizes bounding box to make learning easier
        # converts pixel coordinates to normalized coordinates (0-1)
        # python list -> PyTorch tensor
        cx = (x + w / 2) / W
        cy = (y + h / 2) / H
        bw = w / W
        bh = h / H
        bbox = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)
        
        if self.transform:
            img = self.transform(img)
            
        return img, bbox