from PIL import Image

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