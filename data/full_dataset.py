from torch.utils.data import Dataset
from PIL import Image
import torch
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

# PyTorch requires __init__, __len__, and __getitem__ for dataset classes
class TennisFullDataset(Dataset):
    """
    Loads tennis images from COCO JSON annotations and returns:

        input: exactly like TennisBBoxDataset input side (image with optional transform)
        output: exactly like tennis_pose_dataset output side (label tensor, dtype=torch.long)

    returns
        image: FloatTensor of shape (3, H, W)
        label: Label (0...3) for pose classification
    """
    def __init__(
        self,
        root_dir: str,
        annotation_files: List[str],
        transform=None,
        class_order: Optional[List[str]] = None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Dict] = []

        cat_id_to_name = {}
        json_datas = []

        for ann_path in annotation_files:
            full_ann_path = os.path.join(root_dir, ann_path)
            with Path(full_ann_path).open("r", encoding="utf-8") as f:
                data = json.load(f)

            json_datas.append(data)
            for category in data.get("categories", []):
                cat_id_to_name[category["id"]] = category.get("name", str(category["id"]))

        if not cat_id_to_name:
            raise RuntimeError("No categories found in annotation files.")

        if class_order is not None:
            name_to_id = {name: cid for cid, name in cat_id_to_name.items()}
            missing = [name for name in class_order if name not in name_to_id]
            if missing:
                raise ValueError(f"class_order name not found in categories: {missing}")
            ordered_ids = [name_to_id[name] for name in class_order]
            self.label_names = class_order[:]
        else:
            ordered_ids = sorted(cat_id_to_name.keys())
            self.label_names = [cat_id_to_name[cid] for cid in ordered_ids]

        self.label_map = {cid: i for i, cid in enumerate(ordered_ids)}

        for data in json_datas:
            id_to_img = {img["id"]: img for img in data.get("images", [])}

            for ann in data.get("annotations", []):
                image_id = ann.get("image_id")
                img_info = id_to_img.get(image_id)
                if img_info is None:
                    continue

                cat_id = ann.get("category_id")
                if cat_id not in self.label_map:
                    continue

                rel_path = img_info.get("path", img_info["file_name"])
                rel_path = rel_path.replace("../", "")
                img_path = os.path.join(root_dir, rel_path)

                self.samples.append(
                    {
                        "img_path": img_path,
                        "label": self.label_map[cat_id],
                    }
                )

        if len(self.samples) == 0:
            raise RuntimeError("Loaded 0 samples. Check annotation/image paths and categories.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(sample["label"], dtype=torch.long)
        return img, label
