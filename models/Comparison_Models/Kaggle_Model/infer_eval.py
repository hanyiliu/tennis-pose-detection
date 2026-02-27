# models/Comparison_Models/Kaggle_Model/infer_eval.py

import os
import torch
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torchvision import transforms as T

from models.Comparison_Models.Kaggle_Model.dataset import TennisImageClassDataset


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def run_inference(model, dl, device):
    model.eval()

    y_true, y_pred = [], []
    for x, y in tqdm(dl, desc="Running Inference"):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())

    return np.array(y_true), np.array(y_pred)


def main():
    model_name = "resnet18"
    ckpt_path = "saved_models/tennis_timm_resnet18_160_best.pth"
    im_size = 160
    bs = 32

    device = pick_device()
    print("Using device:", device)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    tfs = T.Compose([
        T.Resize((im_size, im_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # rebuild the same kind of split (stratified) and then evaluate test
    tr_dl, val_dl, test_dl, class_map = TennisImageClassDataset.stratified_split_dls(
        transformations=tfs,
        bs=bs,
        ns=4,
    )

    print("Class Mapping:", class_map)
    num_classes = len(class_map)

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device)

    y_true, y_pred = run_inference(model, test_dl, device)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    inv = {v: k for k, v in class_map.items()}
    names = [inv[i] for i in range(len(inv))]

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=names))

    cm = confusion_matrix(y_true, y_pred)

    os.makedirs("evaluation_outputs", exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix (Test)")
    plt.xticks(range(len(names)), names, rotation=45)
    plt.yticks(range(len(names)), names)
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("evaluation_outputs/confusion_matrix.png")
    plt.close()

    df = pd.DataFrame({"true_label": y_true, "pred_label": y_pred})
    df.to_csv("evaluation_outputs/test_predictions.csv", index=False)

    print("Saved:")
    print("- evaluation_outputs/confusion_matrix.png")
    print("- evaluation_outputs/test_predictions.csv")


if __name__ == "__main__":
    main()