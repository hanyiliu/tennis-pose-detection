from torch.utils.data import DataLoader, random_split
import torch
import os

import argparse
import torch.nn as nn
from torchvision import transforms
from data.bbox_dataset import TennisBBoxDataset
from models.bbox_detection import BBoxDetectionModel
import kagglehub
import os

if os.path.basename(os.getcwd()) == "eda":
    os.chdir("..")

print(f"Current working directory: {os.getcwd()}")

# Set download path to root dir, kagglehub automatically creates datasets subdir
os.environ["KAGGLEHUB_CACHE"] = ""

# Download latest version
path = kagglehub.dataset_download("orvile/tennis-player-actions-dataset")

print("Path to dataset files:", path)

# do we need this idk
# checking which GPU we're using (NVIDIA vs Apple vs other)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # if Apple Silicon GPU via Metal
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
    
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
    parser.add_argument("--root_dir", type=str, required=True, help="...")
    # number of epochs over training set
    parser.add_argument("--epochs", type=int, default=20)
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
    
if __name__ == "__main__":
    main()