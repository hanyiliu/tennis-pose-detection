# train/run_timm_baseline.py

import torch
from torchvision import transforms as T

from models.Comparison_Models.Kaggle_Model.dataset import TennisImageClassDataset
from models.Comparison_Models.Kaggle_Model.train_timm import TrainValidation, TrainConfig


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    dataset_id = "orvile/tennis-player-actions-dataset"

    # ImageNet normalization (works fine for timm pretrained models)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    im_size = 160
    bs = 32
    ns = 4

    tfs = T.Compose([
        T.Resize((im_size, im_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    tr_dl, val_dl, ts_dl, classes = TennisImageClassDataset.stratified_split_dls(
        transformations=tfs,
        dataset_id=dataset_id,
        bs=bs,
        ns=ns,
    )

    device = pick_device()
    print("Using device:", device)
    print("Class map:", classes)

    cfg = TrainConfig(
        model_name="resnet18",
        epochs=8,
        patience=2,
        lr=3e-4,
        save_dir="saved_models",
        save_prefix=f"tennis_timm_resnet18_{im_size}",
        freeze_backbone=False,
        dev_mode=False,
    )

    trainer = TrainValidation(classes=classes, tr_dl=tr_dl, val_dl=val_dl, device=device, cfg=cfg)
    trainer.run()


if __name__ == "__main__":
    main()