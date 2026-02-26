from torch.utils.data import DataLoader, random_split
import torch
import os

import argparse
import torch.nn as nn
from torchvision import transforms
from data.bbox_dataset import TennisBBoxDataset
from models.bbox_detection import BBoxDetectionModel
import kagglehub

if os.path.basename(os.getcwd()) == "eda":
    os.chdir("..")

print(f"Current working directory: {os.getcwd()}")

# Set download path to root dir, kagglehub automatically creates datasets subdir
# os.environ["KAGGLEHUB_CACHE"] = ""

# Download latest version
path = kagglehub.dataset_download("orvile/tennis-player-actions-dataset")
path = os.path.join(
    path,
    "Tennis Player Actions Dataset for Human Pose Estimation"
)

print("Path to dataset files:", path)
print("annotations files:", os.listdir(os.path.join(path, "annotations"))[:5])
print("images subdirs:", os.listdir(os.path.join(path, "images")))

# do we need this idk
# checking which GPU we're using (NVIDIA vs Apple vs other)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # if Apple Silicon GPU via Metal
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def cxcywh_to_xyxy(boxes):
    # input: [min_x, min_y, w, h]
    # output: [x1, y1, x2, y2]
    
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)

# IoU = intersection over union
#    a standard way to measure how well two bboxes overlap
#    (area of overlap) / (area of (prediction OR ground truth))
# A GOOD RESULT (model is performing very well):
#   accuracy@.50: 80-95%
#   accuracy@.75: 40-60%

# A DECENT RESULT (reasonable learning):
#   accuracy@.50: 50-75%
#   accuracy@.75: 20-50%

# A BAD RESULT (no learning, predictions are random):
#   accuracy@.50: 0-10%
#   accuracy@.75: 0%

# eps = epsilon
@torch.no_grad()
def bbox_iou_xyxy(pred, target, eps=1e-7):
    # pred, target in [x1, y1, x2, y2]
    # returns IoU
    
    # ensure proper ordering
    px1 = torch.min(pred[:, 0], pred[:, 2])
    py1 = torch.min(pred[:, 1], pred[:, 3])
    px2 = torch.max(pred[:, 0], pred[:, 2])
    py2 = torch.max(pred[:, 1], pred[:, 3])
    
    tx1 = torch.min(target[:, 0], target[:, 2])
    ty1 = torch.min(target[:, 1], target[:, 3])
    tx2 = torch.max(target[:, 0], target[:, 2])
    ty2 = torch.max(target[:, 1], target[:, 3])
    
    # intersection
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    
    # areas
    p_area = ((px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0))
    t_area = ((tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0))
    
    union = p_area + t_area - inter
    return inter / (union + eps)

@torch.no_grad()
def iou_evaluate(model, loader, criterion, device, iou_thresholds=(0.5, 0.75)):
    model.eval()
    total_loss = 0.0
    n = 0
    
    sum_iou = 0.0
    correct = {thr: 0 for thr in iou_thresholds}
    for imgs, bboxes in loader:
        imgs = imgs.to(device)
        bboxes = bboxes.to(device)
        
        preds = model(imgs)
        loss = criterion(preds, bboxes)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs
        
        # keep normalized coords in range
        preds = preds.clamp(0, 1)
        bboxes = bboxes.clamp(0, 1)
        
        preds_xyxy = cxcywh_to_xyxy(preds)
        bboxes_xyxy = cxcywh_to_xyxy(bboxes)
        
        preds_xyxy = preds_xyxy.clamp(0, 1)
        bboxes_xyxy = bboxes_xyxy.clamp(0, 1)
        
        ious = bbox_iou_xyxy(preds_xyxy, bboxes_xyxy)
        sum_iou += ious.sum().item()
        
        for thr in iou_thresholds:
            correct[thr] += (ious >= thr).sum().item()
    avg_loss = total_loss / max(n, 1)
    mean_iou = sum_iou / max(n , 1)
    acc = {thr: correct[thr] / max(n, 1) for thr in iou_thresholds}
    return avg_loss, mean_iou, acc

@torch.no_grad() # disables gradient tracking for evaluation
# LOL found this out to make eval more efficient
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0 # total number of samples seen so far
    
    for imgs, bboxes in loader:
        imgs = imgs.to(device)
        bboxes = bboxes.to(device)
        
        preds = model(imgs)
        loss = criterion(preds, bboxes)
        
        # bs = batch size
        # computing batch average -> total batch loss ->
        # average of the whole dataset
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs
        
    return total_loss / max(n, 1)

def main():
    parser = argparse.ArgumentParser()
    # where the dataset folder is
    parser.add_argument("--root_dir", type=str, default=path, help="dataset root directory")
    # number of epochs over training set
    parser.add_argument("--epochs", type=int, default=50)
    # how many images per step
    parser.add_argument("--batch_size", type=int, default=16)
    # learning rate
    parser.add_argument("--lr", type=float, default=1e-3)
    # data train test split
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    # makes splits and reproducibility more consistent
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    total = args.train_split + args.val_split + args.test_split

    torch.manual_seed(args.seed)
    
    annotation_files = [
        "annotations/backhand.json",
        "annotations/forehand.json",
        "annotations/ready_position.json",
        "annotations/serve.json",
    ]
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    dataset = TennisBBoxDataset(args.root_dir, annotation_files, transform=transform)
    
    dataset_size = len(dataset)
    train_size = int(dataset_size * args.train_split)
    val_size = int(dataset_size * args.val_split)
    test_size = dataset_size - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
    
    # DataLoader does:
    # batching - returns batches instead of single samples
    # shuffling: randomizes order each epoch (training only)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = get_device()
    print("Device:", device)
    print("Train samples:", len(train_ds), "Val samples:", len(val_ds))
    
    # create model and move it to device
    model = BBoxDetectionModel().to(device)
    
    criterion = nn.SmoothL1Loss()
    # Adam updates weights to reduce loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # shape and correct fc1 size sanity check
    # runs one batch through the odel to catch FC layer input size mismatch
    imgs0, _ = next(iter(train_loader))
    imgs0 = imgs0.to(device)
    try:
        _ = model(imgs0)
    except RuntimeError as e:
        print("\nmodel forward pass failed, fc1 input-size mismatch. fc1 should be nn.Linear(64*32*32, 256)\n")
        raise e
    
    # track best model and creates a checkpoint folder
    best_val = float("inf")
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        
        for imgs, bboxes in train_loader:
            imgs = imgs.to(device)
            bboxes = bboxes.to(device)
            
            # clears old gradients
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, bboxes)
            # computes gradients
            loss.backward()
            # udpates weights
            optimizer.step()
            
            bs = imgs.size(0)
            total_loss += loss.item() * bs
            n += bs
            
        # average loss over all training samples for this epoch
        train_loss = total_loss / max(n, 1)
        # runs model on validation set
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch:02d}/{args.epochs}  train_loss: {train_loss:.5f}  val_loss: {val_loss:.5f}")
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "checkpoints/bbox_best.pt")
    
    print("\n Best val loss:", best_val)
    
    # load best checkpoint before testing
    model.load_state_dict(torch.load("checkpoints/bbox_best.pt", map_location=device))
    test_loss, mean_iou, accuracy = iou_evaluate(model, test_loader, criterion, device)
    
    print("\n Best test loss:", best_val)
    print(f"accuracy@0.50: {accuracy[0.5]*100} %")
    print(f"accuracy@0.75: {accuracy[0.75]*100} %")
    
if __name__ == "__main__":
    main()
    