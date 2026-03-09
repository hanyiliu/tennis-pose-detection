import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from models.bbox_detection import BBoxDetectionModel
import train.bbox_train
import kagglehub
import os

if os.path.basename(os.getcwd()) == "eda":
    os.chdir("..")

print(f"Current working directory: {os.getcwd()}")

# Set download path to root dir, kagglehub automatically creates datasets subdir
os.environ["KAGGLEHUB_CACHE"] = ""

# Download latest version
path = kagglehub.dataset_download("orvile/tennis-player-actions-dataset")
path = os.path.join(
    path,
    "Tennis Player Actions Dataset for Human Pose Estimation"
)

print("Path to dataset files:", path)

# checking which GPU we're using (NVIDIA vs Apple vs other)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # if Apple Silicon GPU via Metal
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# runs bbox model on a single image
# returns pixel bbox [x1, y1, w, h] in the ORIGINAL image coordinate system
# as floats
def infer_bbox(img_path: str, checkpoint_path: str="checkpoints/bbox_best.pt", resize: tuple[int, int]=(384, 384)):

    device = get_device()
    
    # load model + weights
    model = BBoxDetectionModel()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert("RGB")

    W_original, H_original = img.size
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)[0].cpu() # pred contains normalized values
        
    pred = pred.clamp(0, 1)
    cx, cy, bw, bh = pred.tolist()
    x1 = (cx - bw / 2) * W_original
    y1 = (cy - bh / 2) * H_original
    x2 = (cx + bw / 2) * W_original
    y2 = (cy + bh / 2) * H_original
    
    
    x1 = max(0, min(W_original - 1, x1))
    y1 = max(0, min(H_original - 1, y1))
    x2 = max(0, min(W_original - 1, x2))
    y2 = max(0, min(H_original - 1, y2))
    
    w = x2 - x1
    h = y2 - y1

    print("Predicted bounding box:")
    print(f"x1: {x1:.2f}, y1: {y1:.2f}, width: {w:.2f}, height: {h:.2f}")
    
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()

    return [x1, y1, w, h]

if __name__ == "__main__":
    images_dir = os.path.join(path, "images")
    img_path = os.path.join(images_dir, "serve", "S_100.jpeg")
    bbox = infer_bbox(img_path)
    