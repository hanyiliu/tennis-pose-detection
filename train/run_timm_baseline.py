# train/run_timm_baseline.py

# chooses devince, defines preprocessing transformations, gets data loaders and creates training configs, launches training process from train_timm.py

import torch
from torchvision import transforms as T

from models.Comparison_Models.Kaggle_Model.dataset import TennisImageClassDataset
from models.Comparison_Models.Kaggle_Model.train_timm import TrainValidation, TrainConfig


# helper function helps decide which device to use.
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    # stores kaggle ID as string
    dataset_id = "orvile/tennis-player-actions-dataset"

    # ImageNet normalization (works fine for timm pretrained models)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # image size, batch size, number of workers processes can help load images in parellel (faster data loading and reduces wait time)
    im_size = 160
    bs = 32
    num_workers = 4

    # standardized values for image.
    tfs = T.Compose([
        T.Resize((im_size, im_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # preprocessing steps.
    training_dataloader, validation_dataloader, testset_dataloader, classes = TennisImageClassDataset.stratified_split_dls(
        transformations=tfs,
        dataset_id=dataset_id,
        bs=bs,
        num_workers=num_workers,
    )

    device = pick_device()
    print("Using device:", device)
    print("Class map:", classes)

    # creates TrainConfig object, stores all the training settings in one place.
    config = TrainConfig(
        model_name="resnet18",
        epochs=8,
        patience=2,
        lr=3e-4,
        save_dir="saved_models",
        save_prefix=f"tennis_timm_resnet18_{im_size}",
    )
    # creates trainer and start training, creates an instance of the trianing class.
    # passes in, class mapping, training Data loader, validation data loader, device, config settings.
    trainer = TrainValidation(classes=classes, training_dataloader=training_dataloader, validation_dataloader=validation_dataloader, device=device, cfg=config)
    trainer.run()


if __name__ == "__main__":
    main()