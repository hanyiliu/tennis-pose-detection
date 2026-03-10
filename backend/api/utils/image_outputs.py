from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


def create_aggregated_heatmap_image(heatmaps: torch.Tensor) -> Image.Image:
    if heatmaps.dim() != 3:
        raise ValueError(f"Expected (K,H,W) heatmaps; got {tuple(heatmaps.shape)}")

    logits = heatmaps.detach().cpu()
    probs = torch.sigmoid(logits)
    aggregated = probs.sum(dim=0)
    aggregated = aggregated / max(float(aggregated.max().item()), 1e-6)
    image = (aggregated.numpy() * 255).astype(np.uint8)
    return Image.fromarray(image, mode="L").convert("RGB")


def create_keypoint_overlay_image(image: Image.Image, keypoints_norm: torch.Tensor) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    points = keypoints_norm.detach().cpu().numpy()
    for x_norm, y_norm, visibility in points:
        if visibility <= 0:
            continue
        px = int(round(float(x_norm) * (width - 1)))
        py = int(round(float(y_norm) * (height - 1)))
        radius = 3
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(0, 255, 0))

    return canvas


def create_class_prediction_bar_graph(confidences: dict[str, float]) -> Image.Image:
    labels = ["backhand", "forehand", "ready_position", "serve"]
    values = [float(confidences.get(label, 0.0)) for label in labels]

    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    ax.set_title("Pose Class Prediction")
    plt.xticks(rotation=20)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def create_final_overlay_image(
    image: Image.Image,
    bbox_xyxy: tuple[int, int, int, int],
    keypoints_norm: torch.Tensor,
    prediction_label: str,
    confidence: float,
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = bbox_xyxy

    draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
    draw.text((x1, max(y1 - 20, 0)), f"{prediction_label}: {confidence:.2f}", fill=(255, 255, 0))

    points = keypoints_norm.detach().cpu().numpy()
    box_width = max(x2 - x1, 1)
    box_height = max(y2 - y1, 1)
    for x_norm, y_norm, visibility in points:
        if visibility <= 0:
            continue
        px = int(round(x1 + float(x_norm) * box_width))
        py = int(round(y1 + float(y_norm) * box_height))
        radius = 3
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(0, 255, 0))

    return canvas
