# models/Comparison_Models/Kaggle_Model/dataset.py

import os
from glob import glob # finds files by wildcard patterns
from typing import Dict, List, Optional

from PIL import Image # opens images from disk
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import kagglehub


# download dataset (kagglehub caches it after first time)
def resolve_kaggle_root(dataset_id: str) -> str:
    root = kagglehub.dataset_download(dataset_id)

    # sometimes kagglehub wraps everything inside one extra folder
    # so we just check if that's the case and unwrap it
    items = os.listdir(root)

    if len(items) == 1:
        maybe_folder = os.path.join(root, items[0])
        if os.path.isdir(maybe_folder):
            root = maybe_folder

    return root


# get the images folder from the dataset
def find_images_folder(root: str) -> str:
    images_path = os.path.join(root, "images")

    # for this dataset we expect an "images" folder
    # if it's not there, something is wrong
    if not os.path.isdir(images_path):
        raise FileNotFoundError(f"Could not find 'images' folder in {root}")

    return images_path


class TennisImageClassDataset(Dataset):
    # folder structure should look like:
    # images/
    #   forehand/
    #   backhand/
    #   serve/
    #   ready_position/
    def __init__(
        self,
        dataset_id: str = "orvile/tennis-player-actions-dataset",
        root_dir: Optional[str] = None,
        images_root: Optional[str] = None,
        transformations=None,
        image_paths: Optional[List[str]] = None,
        image_labels: Optional[List[int]] = None,
        extensions=(".png", ".jpg", ".jpeg", ".bmp", ".JPG"),
        seed: int = 2025,

    ):
        """
            dataset_id: dataset to download from kaggle
            root_dir: folde rpath where the dataset lives (None automatically downloaded from kaggle, but if given go to local dataset path instead)
            images_root: specific folder containing class folders (backhand, forehand, ready position, serve)
            transformations: resize images, covert to tensor, normalize values. 
            image_paths: speicifc image file paths, used after splitting.
            image_labels: list of numeric labels corresponding to image_paths, keeps label mapping consistent between splits.
            extensions: allows images with the following file extensions
            seed: random seed
        """
    
        # initalize randomneess
        torch.manual_seed(seed)
        self.transformations = transformations

        # if user didn't provide path, download from kaggle
        if root_dir is None:
            root_dir = resolve_kaggle_root(dataset_id)
        # locate images/
        if images_root is None:
            images_root = find_images_folder(root_dir)

        # store both paths.
        self.root_dir = root_dir
        self.images_root = images_root

        # if we already computed splits, just take them
        if image_paths is not None and image_labels is not None: # build train/val/test subset not scanning full dataset.
            # rebuild class mappings and counts
            self.image_paths = image_paths
            self.image_labels = image_labels

            class_names = sorted({self.get_class(p) for p in image_paths}) # go through every image path in split and extract folder name (class label)
            self.class_names: Dict[str, int] = {c: i for i, c in enumerate(class_names)} # creates dictionary mapping, enumerate gives (0, 'backhand) and {c: i ...} gives class_name --> index

            self.class_counts: Dict[str, int] = {} # dictionary to count images per class in split
            # for each image path: get class, increase classe's count, if class exists return current count. if not return 0.
            for p in image_paths: 
                c = self.get_class(p)
                self.class_counts[c] = self.class_counts.get(c, 0) + 1
            return

        # otherwise scan the folders
        paths: List[str] = []
        for ext in extensions:
            paths += glob(os.path.join(images_root, "*", f"*{ext}"))

        # filter any junk if it exists
        self.image_paths = [p for p in paths if "assets" not in p and "scripts" not in p]

        # build class -> id mapping
        self.class_names = {}
        self.class_counts = {}
        # loops thorugh image path and get's class name from this image path.
        for p in self.image_paths:
            c = self.get_class(p)
            # if this is the first time seeing class, assign it to the next avaliable interger id.
            if c not in self.class_names:
                self.class_names[c] = len(self.class_names)
                self.class_counts[c] = 0
            self.class_counts[c] += 1

        #for each image path, get class folder name nad convert class name into integer label so self.im_path[i] matches self.image_labels[i]
        self.image_labels = [self.class_names[self.get_class(p)] for p in self.image_paths]

    # helper function, extract the class label from a file path.
    def get_class(self, path: str) -> str:
        # class name is just the folder name
        return os.path.basename(os.path.dirname(path))

    # helper function: how many items are in the database. (how many batches per epoch)
    def __len__(self):
        return len(self.image_paths)

    # defines what happens when DataLoader asks for one sample.
    def __getitem__(self, index):
        path = self.image_paths[index] 
        y = int(self.image_labels[index]) # gets path and integer label for same index

        img = Image.open(path).convert("RGB") # force it into RGB
        if self.transformations: # prevents issues if an image is grayscale or alpha channel.
            img = self.transformations(img)

        return img, y # returns image_tensor, label_int



    # takes all impages, splits them into balances train/val/test sets returns Dataloaders ready for training and eval.
    @classmethod
    def stratified_split_dls(
        cls, 
        transformations,
        dataset_id: str = "orvile/tennis-player-actions-dataset",
        bs: int = 32,
        split=(0.8, 0.1, 0.1),
        ns: int = 4,
        seed: int = 2025,
    ):
        """
        transformation: preprocessing for images
        dataset_id: kaggle dataset to download/use
        bs: batch size
        split (train %, val %, test %)
        ns: num_worders for dataloader speed
        seed: random seed
        """
        # create dataset.
        ds = cls(dataset_id=dataset_id, transformations=transformations, seed=seed)

        # all file paths and interger labels
        X = ds.image_paths
        y = ds.image_labels

        # first split: train vs temp (val+test)
        train_X, temp_X, train_y, temp_y = train_test_split(
            X, y,
            test_size=(split[1] + split[2]), # test_size=(0.1 + 0.1) = 0.2 = temp gest 20%
            stratify=y, # keeps classes balances in both splits
            random_state=seed
        )

        #Splits temp into val and test while keeping class balance.
        val_ratio = split[1] / (split[1] + split[2])
        val_X, test_X, val_y, test_y = train_test_split(
            temp_X, temp_y,
            test_size=(1 - val_ratio),
            stratify=temp_y,
            random_state=seed
        )

        # train val and test dataset
        train_ds = cls(dataset_id=dataset_id, root_dir=ds.root_dir, images_root=ds.images_root,
                       transformations=transformations, image_paths=train_X, image_labels=train_y, seed=seed)
        val_ds = cls(dataset_id=dataset_id, root_dir=ds.root_dir, images_root=ds.images_root,
                     transformations=transformations, image_paths=val_X, image_labels=val_y, seed=seed)
        test_ds = cls(dataset_id=dataset_id, root_dir=ds.root_dir, images_root=ds.images_root,
                      transformations=transformations, image_paths=test_X, image_labels=test_y, seed=seed)

        # lets dataloader load in parellel, keesp worder processes alive.
        common = dict(num_workers=ns, persistent_workers=(ns > 0))
        if ns > 0:
            common["prefetch_factor"] = 2

        # create dataloaders for train, val, and test
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, **common)
        val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, **common)
        test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, **common)
        
        # return all values from dataloaders and dataset class names mappings to ids. 
        return train_dl, val_dl, test_dl, ds.class_names