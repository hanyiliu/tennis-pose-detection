import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T

from models.Comparison_Models.Kaggle_Model.dataset import TennisImageClassDataset

def main():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    im_size, bs = 160, 16

    tfs = T.Compose([
        T.Resize((im_size, im_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    tr_dl, val_dl, ts_dl, classes = TennisImageClassDataset.stratified_split_dls(
        transformations=tfs,
        bs=bs,
        ns=2,
    )

    inv = {v: k for k, v in classes.items()}
    print("Class map:", classes)
    print("Train size:", len(tr_dl.dataset))
    print("Val size:", len(val_dl.dataset))
    print("Test size:", len(ts_dl.dataset))

    # show one batch
    ims, gts = next(iter(tr_dl))
    # unnormalize for display
    ims_disp = ims.clone()
    for c in range(3):
        ims_disp[:, c] = ims_disp[:, c] * std[c] + mean[c]
    ims_disp = torch.clamp(ims_disp, 0, 1)

    n = min(12, ims_disp.size(0))
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(ims_disp[i].permute(1, 2, 0))
        axes[i].set_title(inv[int(gts[i])])
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()