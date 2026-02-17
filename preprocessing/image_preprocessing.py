import torchvision.transforms as transforms

def convert_image_to_tensor(image):
    """
    Converts a PIL RGB image to a tensor representation.
    Shape: (# of channels, image_height, image_width)
    """
    transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    return transforms(image)