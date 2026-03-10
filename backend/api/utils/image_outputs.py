from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _load_label_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_name in ["DejaVuSans.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _figure_to_pil(fig) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def create_aggregated_heatmap_image(heatmaps: torch.Tensor) -> Image.Image:
    if heatmaps.dim() != 3:
        raise ValueError(f"Expected (K,H,W) heatmaps; got {tuple(heatmaps.shape)}")

    logits = heatmaps.detach().cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    summed_heatmap = probs.sum(axis=0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(summed_heatmap, cmap="inferno")
    ax.set_title("All Keypoint Heatmaps (Sum)")
    ax.axis("off")
    fig.tight_layout()

    return _figure_to_pil(fig)


def create_keypoint_overlay_image(
    image: Image.Image,
    keypoints_xyv: torch.Tensor,
    heatmap_size: tuple[int, int],
) -> Image.Image:
    points = keypoints_xyv.detach().cpu().numpy()
    img_w, img_h = image.size

    hmap_h, hmap_w = heatmap_size

    scale_x = (img_w - 1) / max(hmap_w - 1, 1)
    scale_y = (img_h - 1) / max(hmap_h - 1, 1)
    pred_x = np.clip(points[:, 0] * scale_x, 0, img_w - 1)
    pred_y = np.clip(points[:, 1] * scale_y, 0, img_h - 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.scatter(pred_x, pred_y, c="cyan", s=35, edgecolors="black", linewidths=0.6)
    for idx, (xk, yk, _vis) in enumerate(points):
        ax.text((xk * scale_x) + 1.5, (yk * scale_y) + 1.5, str(idx), color="yellow", fontsize=7)
    ax.set_title("Stage 2 Keypoint Predictions (Argmax)")
    ax.axis("off")
    fig.tight_layout()

    return _figure_to_pil(fig)


def create_class_prediction_bar_graph(confidences: dict[str, float]) -> Image.Image:
    labels = ["backhand", "forehand", "ready_position", "serve"]
    values = [float(confidences.get(label, 0.0)) for label in labels]

    fig, ax = plt.subplots(figsize=(7, 3), dpi=100)
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    ax.set_title("Pose Class Probabilities")
    fig.tight_layout()

    return _figure_to_pil(fig)


def _letterbox_xy_to_bbox_xy(
    x_letterbox: float,
    y_letterbox: float,
    bbox_xyxy: tuple[int, int, int, int],
    model_image_size: tuple[int, int],
) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    bbox_w = float(max(x2 - x1, 1))
    bbox_h = float(max(y2 - y1, 1))

    out_h, out_w = model_image_size
    scale = min(out_w / bbox_w, out_h / bbox_h)
    resized_w = bbox_w * scale
    resized_h = bbox_h * scale
    pad_x = (out_w - resized_w) / 2.0
    pad_y = (out_h - resized_h) / 2.0

    x_crop = (x_letterbox - pad_x) / max(scale, 1e-6)
    y_crop = (y_letterbox - pad_y) / max(scale, 1e-6)
    x_crop = float(np.clip(x_crop, 0.0, bbox_w - 1.0))
    y_crop = float(np.clip(y_crop, 0.0, bbox_h - 1.0))

    return x1 + x_crop, y1 + y_crop


def create_final_overlay_image(
    image: Image.Image,
    bbox_xyxy: tuple[int, int, int, int],
    keypoints_xyv: torch.Tensor,
    keypoint_image_size: tuple[int, int],
    prediction_label: str,
    confidence: float,
) -> Image.Image:
    canvas = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = bbox_xyxy

    draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)

    label_text = prediction_label.replace("_", " ").title()
    confidence_text = f"Confidence: {confidence:.2f}"
    text = f"{label_text} | {confidence_text}"

    text_x = 12
    text_h = 72
    text_w = max(660, (len(text) * 24) + 60)
    text_y = max(canvas.height - text_h - 12, 0)
    text_x2 = min(text_x + text_w, canvas.width - 12)

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        (text_x, text_y, text_x2, text_y + text_h),
        fill=(0, 0, 0, 140),
    )
    canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas)
    label_font = _load_label_font(size=36)
    draw.text(
        (text_x + 20, text_y + 18),
        text,
        fill=(255, 255, 255, 255),
        font=label_font,
    )

    points = keypoints_xyv.detach().cpu().numpy()
    for x_letterbox, y_letterbox, visibility in points:
        if visibility <= 0:
            continue
        px_float, py_float = _letterbox_xy_to_bbox_xy(
            x_letterbox=x_letterbox,
            y_letterbox=y_letterbox,
            bbox_xyxy=bbox_xyxy,
            model_image_size=keypoint_image_size,
        )
        px = int(round(px_float))
        py = int(round(py_float))
        radius = 3
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(0, 255, 0))

    return canvas.convert("RGB")
