import torchvision.transforms as transform

def convert_image_to_tensor(image):
    """
    Converts a PIL RGB image to a tensor representation.
    Shape: (# of channels, image_height, image_width)
    """
    transform = transform.Compose([
        transform.ToTensor(),
    ])

    return transform(image)