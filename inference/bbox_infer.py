import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from models.bbox_detection import BBoxDetectionModel
import train.bbox_train

# checking which GPU we're using (NVIDIA vs Apple vs other)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # if Apple Silicon GPU via Metal
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# runs bbox model on a single image
# returns pixel bbox [x, y, w, h] in the ORIGINAL image coordinate system
# as floats
def infer_bbox(img_path: str, checkpoint_path: str="checkpoints/bbox_best.pt", resize: tuple[int, int]=(256, 256)):

    device = get_device()
    
    # load model + weights
    model = BBoxDetectionModel()
    model.load_state_dict(torch.load("checkpoints/bbox_best.pt", map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])

    img = Image.open(img_path).convert("RGB")

    W_original, H_original = img.size
    img_tensor = transform(img).unsqueeze(0).to(device) # (1, 3, 256, 256)

    with torch.no_grad():
        pred = model(img_tensor)[0].cpu() # pred contains normalized values
        
    x = pred[0].item() * W_original
    y = pred[1].item() * H_original
    w = pred[2].item() * W_original
    h = pred[3].item() * H_original

    print("Predicted bounding box:")
    print(f"x_min: {x:.2f}, y_min: {y:.2f}, width: {w:.2f}, height: {h:.2f}")
    
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()

    return [x, y, w, h]

if __name__ == "__main__":
    img_path = "images/backhand/B_001.jpeg"
    bbox = infer_bbox(img_path)
    