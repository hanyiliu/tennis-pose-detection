from torch.utils.data import DataLoader
import torch
import os

import argparse
import torch.nn as nn
from torchvision import transforms
from data.bbox_dataset import TennisBBoxDataset
from models.bbox_detection import BBoxDetectionModel
import kagglehub
from torch.utils.data import Subset

from torchvision.ops import complete_box_iou_loss

if os.path.basename(os.getcwd()) == "eda":
    os.chdir("..")

# Set download path to root dir, kagglehub automatically creates datasets subdir
# os.environ["KAGGLEHUB_CACHE"] = ""

# Download latest version
path = kagglehub.dataset_download("orvile/tennis-player-actions-dataset")
path = os.path.join(
    path,
    "Tennis Player Actions Dataset for Human Pose Estimation"
)

# checking which GPU we're using (NVIDIA vs Apple vs other)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # if Apple Silicon GPU via Metal
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def xywh_to_xyxy(boxes):
    # input: [min_x, min_y, w, h]
    # output: [x1, y1, x2, y2]
    
    min_x, min_y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = min_x
    y1 = min_y
    x2 = min_x + w
    y2 = min_y + h
    return torch.stack([x1, y1, x2, y2], dim=1)

# validates bbox dimensions
def sanitize_xyxy(boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    boxes = boxes.clamp(0, 1)

    x1 = torch.min(boxes[:, 0], boxes[:, 2])
    y1 = torch.min(boxes[:, 1], boxes[:, 3])
    x2 = torch.max(boxes[:, 0], boxes[:, 2])
    y2 = torch.max(boxes[:, 1], boxes[:, 3])

    # avoid zero-width / zero-height boxes
    x2 = torch.maximum(x2, x1 + eps)
    y2 = torch.maximum(y2, y1 + eps)

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

def ciou_loss_xywh(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # IoU loss = mean(1 - IoU), where IoU is computed after converting cxcywh -> xyxy
    preds_xyxy = sanitize_xyxy(xywh_to_xyxy(preds))
    targets_xyxy = sanitize_xyxy(xywh_to_xyxy(targets))
    return complete_box_iou_loss(preds_xyxy, targets_xyxy, reduction="mean")

@torch.no_grad() # disables gradient tracking for evaluation
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
        loss_reg = criterion(preds, bboxes)
        loss_ciou = ciou_loss_xywh(preds, bboxes)
        loss = loss_reg + 0.5 * loss_ciou
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs
        
        # keep normalized coords in range
        preds = preds.clamp(0, 1)
        bboxes = bboxes.clamp(0, 1)
        
        preds_xyxy = sanitize_xyxy(xywh_to_xyxy(preds))
        bboxes_xyxy = sanitize_xyxy(xywh_to_xyxy(bboxes))
        
        ious = bbox_iou_xyxy(preds_xyxy, bboxes_xyxy)
        sum_iou += ious.sum().item()
        
        for thr in iou_thresholds:
            correct[thr] += (ious >= thr).sum().item()
    avg_loss = total_loss / max(n, 1)
    mean_iou = sum_iou / max(n , 1)
    acc = {thr: correct[thr] / max(n, 1) for thr in iou_thresholds}
    return avg_loss, mean_iou, acc

def main():
    parser = argparse.ArgumentParser()
    # where the dataset folder is
    parser.add_argument("--root_dir", type=str, default=path, help="dataset root directory")
    # number of epochs over training set
    parser.add_argument("--epochs", type=int, default=50)
    # how many images per step
    parser.add_argument("--batch_size", type=int, default=32)
    # learning rate
    parser.add_argument("--lr", type=float, default=3e-4)
    # data train test split
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    # makes splits and reproducibility more consistent
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iou_lambda", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    annotation_files = [
        "annotations/backhand.json",
        "annotations/forehand.json",
        "annotations/ready_position.json",
        "annotations/serve.json",
    ]
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    train_dataset = TennisBBoxDataset(args.root_dir, annotation_files, transform=train_transform)
    eval_dataset  = TennisBBoxDataset(args.root_dir, annotation_files, transform=eval_transform)

    dataset_size = len(train_dataset)
    train_size = int(dataset_size * args.train_split)
    val_size   = int(dataset_size * args.val_split)
    test_size  = dataset_size - train_size - val_size

    # split indices, not dataset objects, for reproducibility
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(dataset_size, generator=g).tolist()

    train_idx = perm[:train_size]
    val_idx   = perm[train_size:train_size + val_size]
    test_idx  = perm[train_size + val_size:]

    train_ds = Subset(train_dataset, train_idx)
    val_ds   = Subset(eval_dataset,  val_idx)
    test_ds  = Subset(eval_dataset,  test_idx)
    
    # DataLoader does:
    # batching- returns batches instead of single samples
    # shuffling- randomizes order each epoch (training only)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
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
    best_val_mean_iou = -1.0
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
            loss_reg = criterion(preds, bboxes)
            loss_ciou = ciou_loss_xywh(preds, bboxes)
            loss = loss_reg + args.iou_lambda * loss_ciou
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
        val_loss, val_mean_iou, val_acc = iou_evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{args.epochs}  train_loss: {train_loss:.5f}  val_loss: {val_loss:.5f}  val_acc@0.50: {val_acc[0.5]*100:.2f}% val_acc@0.75: {val_acc[0.75]*100:.2f}%")
        
        if best_val_mean_iou < val_mean_iou:
            best_val_mean_iou = val_mean_iou
            torch.save(model.state_dict(), "checkpoints/bbox_best.pt")
    
    # load best checkpoint before testing
    model.load_state_dict(torch.load("checkpoints/bbox_best.pt", map_location=device))
    test_loss, test_mean_iou, test_acc = iou_evaluate(model, test_loader, criterion, device)
    
    print("\n Test loss:", test_loss)
    print("\n Test mean IoU:", test_mean_iou)
    print(f"accuracy@0.50: {test_acc[0.5]*100} %")
    print(f"accuracy@0.75: {test_acc[0.75]*100} %")
    
if __name__ == "__main__":
    main()
    