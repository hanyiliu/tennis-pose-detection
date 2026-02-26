# models/Comparison_Models/Kaggle_Model/dataset.py

import os
from glob import glob
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import kagglehub


def _resolve_kaggle_root(dataset_id: str) -> str:
    """
    Download (or reuse cached) Kaggle dataset and return the usable root directory.
    KaggleHub often returns a directory that contains exactly one nested folder;
    we unwrap that like you did in pose_dataset.py.
    """
    root_dir = kagglehub.dataset_download(dataset_id)

    subdirs = os.listdir(root_dir)
    if len(subdirs) == 1:
        possible_root = os.path.join(root_dir, subdirs[0])
        if os.path.isdir(possible_root):
            root_dir = possible_root

    return root_dir


def _find_images_root(root_dir: str) -> str:
    """
    Find the folder that contains class subfolders of images.
    We look for a directory named 'images' first; otherwise search recursively.
    """
    # common case
    direct = os.path.join(root_dir, "images")
    if os.path.isdir(direct):
        return direct

    # search for any folder named images
    for dirpath, dirnames, _ in os.walk(root_dir):
        if "images" in dirnames:
            return os.path.join(dirpath, "images")

    raise FileNotFoundError(f"Could not find an 'images' folder under: {root_dir}")


class TennisImageClassDataset(Dataset):
    """
    Image classification dataset where class label = folder name under images_root:
        images_root/
            backhand/
            forehand/
            serve/
            ready/
    Returns (image_tensor, label_int)
    """

    def __init__(
        self,
        dataset_id: str = "orvile/tennis-player-actions-dataset",
        root_dir: Optional[str] = None,
        images_root: Optional[str] = None,
        transformations=None,
        im_paths: Optional[List[str]] = None,
        im_lbls: Optional[List[int]] = None,
        im_files=(".png", ".jpg", ".jpeg", ".bmp", ".JPG"),
        seed: int = 2025,
    ):
        torch.manual_seed(seed)

        # Resolve root via KaggleHub (cached) unless caller provided local root_dir
        if not root_dir:
            root_dir = _resolve_kaggle_root(dataset_id)

        # Find images root unless provided
        if not images_root:
            images_root = _find_images_root(root_dir)

        self.root_dir = root_dir
        self.images_root = images_root
        self.transformations = transformations

        # If split provided, just use those
        if im_paths is not None and im_lbls is not None:
            self.im_paths = im_paths
            self.im_lbls = im_lbls
            # build class map from existing paths
            class_names = sorted(list({self.get_class(p) for p in self.im_paths}))
            self.cls_names: Dict[str, int] = {c: i for i, c in enumerate(class_names)}
            self.cls_counts: Dict[str, int] = {}
            for p in self.im_paths:
                c = self.get_class(p)
                self.cls_counts[c] = self.cls_counts.get(c, 0) + 1
            return

        # Otherwise scan images_root/*/*.{ext}
        all_paths: List[str] = []
        for ext in im_files:
            all_paths.extend(glob(os.path.join(images_root, "*", f"*{ext}")))

        # (Optional) filter out unwanted folders if any exist
        self.im_paths = [
            p for p in all_paths
            if ("assets" not in p) and ("scripts" not in p)
        ]

        # Build label mapping from folder names
        self.cls_names = {}
        self.cls_counts = {}
        for p in self.im_paths:
            c = self.get_class(p)
            if c not in self.cls_names:
                self.cls_names[c] = len(self.cls_names)
                self.cls_counts[c] = 0
            self.cls_counts[c] += 1

        self.im_lbls = [self.cls_names[self.get_class(p)] for p in self.im_paths]

    def get_class(self, path: str) -> str:
        return os.path.basename(os.path.dirname(path))

    def __len__(self) -> int:
        return len(self.im_paths)

    def __getitem__(self, idx: int):
        im_path = self.im_paths[idx]
        im = Image.open(im_path).convert("RGB")
        gt = int(self.im_lbls[idx])
        if self.transformations is not None:
            im = self.transformations(im)
        return im, gt

    @classmethod
    def stratified_split_dls(
        cls,
        transformations,
        dataset_id: str = "orvile/tennis-player-actions-dataset",
        root_dir: Optional[str] = None,
        images_root: Optional[str] = None,
        bs: int = 16,
        split=(0.8, 0.1, 0.1),
        ns: int = 4,
        seed: int = 2025,
    ):
        dataset = cls(
            dataset_id=dataset_id,
            root_dir=root_dir,
            images_root=images_root,
            transformations=transformations,
            seed=seed,
        )

        im_paths = dataset.im_paths
        labels = dataset.im_lbls

        train_paths, temp_paths, train_lbls, temp_lbls = train_test_split(
            im_paths,
            labels,
            test_size=(split[1] + split[2]),
            stratify=labels,
            random_state=seed,
        )

        val_ratio = split[1] / (split[1] + split[2])
        val_paths, test_paths, val_lbls, test_lbls = train_test_split(
            temp_paths,
            temp_lbls,
            test_size=(1 - val_ratio),
            stratify=temp_lbls,
            random_state=seed,
        )

        train_ds = cls(dataset_id=dataset_id, root_dir=dataset.root_dir, images_root=dataset.images_root,
                       transformations=transformations, im_paths=train_paths, im_lbls=train_lbls, seed=seed)
        val_ds   = cls(dataset_id=dataset_id, root_dir=dataset.root_dir, images_root=dataset.images_root,
                       transformations=transformations, im_paths=val_paths,   im_lbls=val_lbls, seed=seed)
        test_ds  = cls(dataset_id=dataset_id, root_dir=dataset.root_dir, images_root=dataset.images_root,
                       transformations=transformations, im_paths=test_paths,  im_lbls=test_lbls, seed=seed)

        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=ns)
        val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=ns)
        test_dl  = DataLoader(test_ds,  batch_size=1,  shuffle=False, num_workers=ns)

        return train_dl, val_dl, test_dl, dataset.cls_names