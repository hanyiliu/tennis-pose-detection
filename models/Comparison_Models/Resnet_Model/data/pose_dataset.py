from PIL import Image
from torch.utils.data import Dataset


class TennisPoseDataset(Dataset):
    """
    each sample is (img_path, class_name)

    returns:
        image: tensor of shape (3, H, W)
        label: int in {0, 1, 2, 3}
    """

    class_to_idx = {
        "forehand": 0,
        "backhand": 1,
        "ready_position": 2,
        "serve": 3,
    }

    idx_to_class = {
        0: "forehand",
        1: "backhand",
        2: "ready_position",
        3: "serve",
    }

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]

        # load image
        image = Image.open(img_path).convert("RGB")

        # convert class name to integer label
        label = self.class_to_idx[class_name]

        # apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, label