# train/pose_dataset.py

import os
import json
from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

import kagglehub

class tennis_pose_dataset(Dataset):
    """
    Load COCO keypoints from annotations and returns:
        x: (18,3) tensor [x_norm, y_norm, visibility]
        y: label (0...3) for pose classification
    """

    def __init__(
        self,
        root_dir: str, # directory where annotation files are located
        annotation_files: List[str], # list of annotation file names (relative to root_dir)
        noramlize_xy: bool = True, # if True, normalize x,y to [0,1] based on image dimensions
        class_order: Optional[List[str]] = None, # optional list of class names in order corresponding to labels 0...3
                                                 # if None, will infer from annotation data
    ):
        # If root_dir is not provided, download (or reuse cached) dataset the same way bbox_train does
        if root_dir is None or root_dir == "":
            root_dir = kagglehub.dataset_download("orvile/tennis-player-actions-dataset")

            # handle extra nested folder (KaggleHub layout)
            subdirs = os.listdir(root_dir)
            if len(subdirs) == 1:
                possible_root = os.path.join(root_dir, subdirs[0])
                if os.path.isdir(possible_root):
                    root_dir = possible_root

            print("Path to dataset files:", root_dir)

        # list will hold everything needed for training (keypoints, width, height, label)
        self.samples = []

        # collect categories across all JSONs: {id: name}
        cat_id_to_name = {} # maps category id to name: 0 --> Backhand
        json_datas = [] # to hold loaded JSON data for processing after we have the category mapping (since category mapping is needed to assign labels to samples)
        for rel_path in annotation_files: # loop through each json file
            full_path = os.path.join(root_dir, rel_path) # construct full path to json file by joining root_dir and relative path
            with open(full_path, "r") as f:
                data = json.load(f) # load json data from file, this should give us a dictionary with keys like "images", "annotations", and "categories" based on COCO format. We will process this data to extract keypoints and labels for our dataset.
            json_datas.append(data) # store loaded json data for later processing after we have the category mapping
            for c in data.get("categories", []): # loop through categories in this json file and add to our category mapping. This allows us to build a complete mapping of category ids to names across all annotation files, which is necessary for assigning consistent labels to our samples.
                cat_id_to_name[c["id"]] = c.get("name", str(c["id"]))

        # error if no categories found in any JSON file
        if not cat_id_to_name:
            raise RuntimeError("No categories found in annotation files.")

        # build mapping category_id -> label index (0...3)
        if class_order is not None:
            # map by name (best for reporting consistent labels across different annotation files)
            name_to_id = {name: cid for cid, name in cat_id_to_name.items()} # reverse mapping from category name to id, e.g. "Backhand" --> 0
            missing = [n for n in class_order if n not in name_to_id] # check if all class names in class_order are present in the categories we found in the JSON files. If any are missing, raise an error to alert the user that their specified class order is invalid.
            if missing:
                raise ValueError(f"class_order name not found in categories: {missing}")
            ordered_ids = [name_to_id[n] for n in class_order] # build ordered_ids in the ordering we want
            self.label_names = class_order[:] # stores label names to print later
        else:
            #default by sorted category_id
            ordered_ids = sorted(cat_id_to_name.keys()) # if order is not provided
            self.label_names = [cat_id_to_name[cid] for cid in ordered_ids] # store label names in the order corresponding to their assigned label indices (0...3) for potential use in reporting or analysis. This allows us to know which label index corresponds to which pose class name.

        self.label_map = {cid: i for i, cid in enumerate(ordered_ids)}  # category id --> class index (0..3), this is what we will use to assign labels to our samples based on their category ids in the annotations.
        self.normalize_xy = noramlize_xy # whether to divide x,y by width/height in __getitem__ 

        # build sample
        for data in json_datas: # create mapping image_id --> image info so we can look up image width/height
            id_to_img = {img["id"]: img for img in data.get("images", [])}
 
            for ann in data.get("annotations", []): # loop through annotations in json file, get keypoints lists, skip if not a list or not length 54 (18 * 3).
                keypoints = ann.get("keypoints", None)
                if not isinstance(keypoints, list) or len(keypoints) != 54:
                    continue
                    
                img_info = id_to_img.get(ann["image_id"], {}) # find image width/heigh for that annocation's image_id
                width = img_info.get("width", 1)
                height = img_info.get("height", 1)
                cat_id = ann.get("category_id", None)

                if cat_id not in self.label_map: # if category id doesn't match your chosen classes, skip.
                    continue

                self.samples.append((keypoints, width, height, self.label_map[cat_id])) # store training example.

        if len(self.samples) == 0: # ensure we loaded something
            raise RuntimeError("Loaded 0 samples. Check JSON keypoints format")
    
    def __len__(self): # get number of samples.
        return len(self.samples)
    
    def __getitem__(self, index): #loads one stored sample
        keypoints, width, height, label = self.samples[index]
        keypoints = np.array(keypoints, dtype=np.float32).reshape(18,3) # turns flat list of 54 into (18, 3), where 18 is number of keypoints and 3 is (x,y,visibility) for each keypoint.

        if self.normalize_xy:
            width = max(float(width), 1.0)
            height = max(float(height), 1.0)
            keypoints[:, 0] /= width # normalize x
            keypoints[:, 1] /= height # normalize y

        return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(label, dtype=torch.long) # returns keypoints as (18,3) tensor and label as scalar tensor for this sample
