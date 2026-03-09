# models/Comparison_Models/Kaggle_Model/infer_eval.py

# reloads the trained model, runs it on test set, prints metrics and analysis, saves confusion matrix and predictions CSV.

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

# helper function to pick device for hardware to use.
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# donot use gradients inside this function.
@torch.no_grad()
# runs model on a data loader and collects predictions.
def run_inference(model, dl, device):
    model.eval()
    # creates lists that stores actual labels and predicted labels.
    y_true, y_pred = [], []
    # loops over the data loaders batch by batch.
    for x, y in tqdm(dl, desc="Running Inference"):
        x = x.to(device)
        y = y.to(device)

        # forward pass thorugh network.
        logits = model(x)
        pred = logits.argmax(dim=1)

        # finds index of the largest score for each image and becomes predicted class id. (ermmmmm...?)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())

    return np.array(y_true), np.array(y_pred)


def main():
    # specifies architecture to rebuild
    # specifies file path of the saved best model weights.
    # image size for preprocessing
    #batch size for rebuilding data loaders
    model_name = "resnet18"
    ckpt_path = "saved_models/tennis_timm_resnet18_160_best.pth"
    im_size = 160
    bs = 32

    # picks devices
    device = pick_device()
    print("Using device:", device)

    # normalize values from ImageNet. 
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # transform pipline, resize images, 
    # converts image from PIL format to Pytorch tensor, 
    # normalizes image
    tfs = T.Compose([
        T.Resize((im_size, im_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # rebuild the same kind of split (stratified) and then evaluate test. 
    tr_dl, val_dl, test_dl, class_map = TennisImageClassDataset.stratified_split_dls(
        transformations=tfs,
        bs=bs,
        ns=4,
    )
    
    # prints class-to-id mapping to verify label order
    print("Class Mapping:", class_map)
    num_classes = len(class_map)

    # recreate the same model architecture as training from our own trained weights.
    # loads teh saved learned weights from disk to model
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device)

    # runs inference on test data loader
    y_true, y_pred = run_inference(model, test_dl, device)

    # computes metrics: accuracy and macro F1
    # (macro F1 calcualtes unweighted averages of F1 scores for each class, treats all classes equally regardless of frequency.)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    # prints results
    print("\n===== TEST RESULTS =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    # reverses class map and builds a list of class names in numeric order.
    inverse = {v: k for k, v in class_map.items()}
    names = [inverse[i] for i in range(len(inverse))]


    # print report and confusion matrix
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