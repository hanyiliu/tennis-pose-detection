from PIL import Image


def norm_bbox_to_xyxy_pixels(bbox_pred, image_width, image_height):
    """Convert normalized bbox (x, y, w, h) to (x1, y1, x2, y2).

    Args:
        bbox_pred (torch.tensor): Normalized bbox
        image_width (int): Width of image in pixels
        image_height (int): Height of image in pixels

    Raises:
        ValueError: Incorrect bbox prediction size.

    Returns:
        torch.tensor: XYXY unnormalized pixel bounding box.
    """    
    bbox = bbox_pred.detach().cpu().view(-1)
    if bbox.numel() != 4:
        raise ValueError(f"Expected bbox prediction with 4 values, got shape {tuple(bbox_pred.shape)}")

    x, y, w, h = bbox.tolist()
    x = x * image_width
    y = y * image_height
    w = w * image_width
    h = h * image_height

    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))
    return x1, y1, x2, y2

def crop_pil(image, bbox):
    """
    Crops the PIL image based on a bounding box [x_min, y_min, x_max, y_max].
    """
    return image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

def letterbox_resize(image, size):
    """
    Resizes PIL image to size (new_height, new_width) using letterboxing to maintain aspect ratio.
    """
    new_height, new_width = size
    image.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
    
    # create new black background image
    new_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    # paste resized image in center
    upper = (new_height - image.size[1]) // 2
    left = (new_width - image.size[0]) // 2
    new_image.paste(image, (left, upper))
    
    return new_image