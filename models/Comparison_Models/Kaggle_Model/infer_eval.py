# models/Comparison_Models/Kaggle_Model/infer_eval.py

# reloads the trained model, runs it on test set, prints metrics and analysis, saves confusion matrix and predictions CSV.

import os
import torch
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    # rebuild the same stratified dataset split used during training
    training_dataloader, validation_dataloader, test_dataloader, class_map = TennisImageClassDataset.stratified_split_dls(
        transformations=tfs,
        bs=bs,
        num_workers=4,
    )

    print("Train size:", len(training_dataloader.dataset))
    print("Val size:", len(validation_dataloader.dataset))
    print("Test size:", len(test_dataloader.dataset))
    
    # prints class-to-id mapping to verify label order
    print("Class Mapping:", class_map)
    num_classes = len(class_map)

    # recreate the same model architecture as training from our own trained weights.
    # loads teh saved learned weights from disk to model
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device)

    # runs inference on test data loader
    y_true, y_pred = run_inference(model, test_dataloader, device)

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

    # builds confusion matrix using true labels and predicted labels.
    cm = confusion_matrix(y_true, y_pred)

    # makes sure output folder exists before saving the graphs and csv.
    os.makedirs("evaluation_outputs", exist_ok=True)

    # creates a cleaner heatmap style confusion matrix.
    # annot=True writes the counts in each square.
    # fmt="d" makes the values show as integers.
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names
    )

    # titles and axis labels for confusion matrix graph.
    plt.title("Confusion Matrix (Predicted vs Actual)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # makes spacing cleaner so labels do not overlap.
    plt.tight_layout()

    # saves confusion matrix graph.
    plt.savefig("evaluation_outputs/confusion_matrix.png")
    plt.close()

    print("Confusion matrix saved to evaluation_outputs/confusion_matrix.png")

    # gets full classification report as a dictionary so we can grab class f1-scores.
    report_dict = classification_report(y_true, y_pred, target_names=names, output_dict=True)

    # collects per-class f1 scores in same order as class names.
    per_class_f1 = [report_dict[class_name]["f1-score"] for class_name in names]

    # creates bar chart for per-class f1 scores.
    plt.figure(figsize=(7, 5))
    plt.bar(names, per_class_f1)

    # graph title and labels.
    plt.title("Per-Class F1 Score")
    plt.xlabel("Tennis Action Class")
    plt.ylabel("F1 Score")

    # f1 score ranges from 0 to 1, so keep y-axis in that range.
    plt.ylim(0, 1.0)

    # rotates x labels so they are easier to read.
    plt.xticks(rotation=25)

    # writes exact f1 score above each bar.
    for i, score in enumerate(per_class_f1):
        plt.text(i, score + 0.02, f"{score:.2f}", ha="center")

    # makes layout cleaner.
    plt.tight_layout()

    # saves f1 bar chart.
    plt.savefig("evaluation_outputs/per_class_f1_scores.png")
    plt.close()

    print("Per-class F1 chart saved to evaluation_outputs/per_class_f1_scores.png")

    # saves predictions into csv file for later analysis if needed.
    df = pd.DataFrame({"true_label": y_true, "pred_label": y_pred})
    df.to_csv("evaluation_outputs/test_predictions.csv", index=False)

    print("Saved:")
    print("- evaluation_outputs/confusion_matrix.png")
    print("- evaluation_outputs/per_class_f1_scores.png")
    print("- evaluation_outputs/test_predictions.csv")


if __name__ == "__main__":
    main()