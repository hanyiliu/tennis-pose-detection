import torchvision.transforms as torch_transforms

def convert_image_to_tensor(image):
    """
    Converts a PIL RGB image to a tensor representation.
    Shape: (# of channels, image_height, image_width)
    """
    transform = torch_transforms.Compose([
        torch_transforms.ToTensor(),
    ])

    return transform(image)