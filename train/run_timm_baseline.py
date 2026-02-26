# train/run_timm_baseline.py

import torch
from torchvision import transforms as T

from models.Comparison_Models.Kaggle_Model.dataset import TennisImageClassDataset
from models.Comparison_Models.Kaggle_Model.train_timm import TrainValidation, TrainConfig


def main():
    dataset_id = "orvile/tennis-player-actions-dataset"  # kagglehub id

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    im_size, bs = 224, 16

    tfs = T.Compose([
        T.Resize((im_size, im_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    tr_dl, val_dl, ts_dl, classes = TennisImageClassDataset.stratified_split_dls(
        transformations=tfs,
        dataset_id=dataset_id,
        bs=bs,
        ns=4,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = TrainConfig(
        model_name="rexnet_150",
        save_dir="saved_models",
        save_prefix="tennis_timm_rexnet150",
        epochs=30,
        patience=3,
        threshold=0.0,
    )

    trainer = TrainValidation(classes=classes, tr_dl=tr_dl, val_dl=val_dl, device=device, cfg=cfg)
    trainer.run()


if __name__ == "__main__":
    main()