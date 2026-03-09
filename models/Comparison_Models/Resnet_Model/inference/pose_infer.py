import os
import torch
from PIL import Image
import kagglehub
from data.transforms import get_eval_transforms
from models.pose_classification import TennisPoseClassifier

IMG_SIZE = 224
CHECKPOINT_PATH = "best_model.pt"
CLASS_NAMES = ["forehand", "backhand", "ready_position", "serve"]

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_dataset_root():
    dataset_path = kagglehub.dataset_download("orvile/tennis-player-actions-dataset")
    dataset_root = os.path.join(
        dataset_path,
        "Tennis Player Actions Dataset for Human Pose Estimation"
    )
    return dataset_root

def predict_image(image_path):
    device = get_device()

    model = TennisPoseClassifier(num_classes=4, pretrained=False).to(device)
    model.load_state_dict(CHECKPOINT_PATH, model, map_location=device)
    model.eval()

    transform = get_eval_transforms(IMG_SIZE)
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    return CLASS_NAMES[pred_idx], probs.squeeze(0).cpu().tolist()


if __name__ == "__main__":
    dataset_root = get_dataset_root()
    img_path = os.path.join(dataset_root, "images", "forehand", "F_067.jpeg")
    pred_class, probs = predict_image(img_path)
    print("Prediction:", pred_class)
    print("Probabilities:", probs)