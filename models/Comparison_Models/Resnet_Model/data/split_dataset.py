import os
import random
from collections import defaultdict
import kagglehub

def get_dataset_root():

    dataset_path = kagglehub.dataset_download("orvile/tennis-player-actions-dataset")
    dataset_root = os.path.join(
        dataset_path,
        "Tennis Player Actions Dataset for Human Pose Estimation"
    )
    return dataset_root


def get_images_root():
    dataset_root = get_dataset_root()
    images_root = os.path.join(dataset_root, "images")
    return images_root

def build_samples_from_folders(images_root=None):
    """
    expects:
    images/
      forehand/
      backhand/
      ready_position/
      serve/
    """
    if images_root is None:
        images_root = get_images_root()
        
    class_names = ["forehand", "backhand", "ready_position", "serve"]
    samples = []

    for class_name in class_names:
        class_dir = os.path.join(images_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpeg")):
                samples.append((os.path.join(class_dir, fname), class_name))

    return samples


def stratified_split(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)
    grouped = defaultdict(list)

    for sample in samples:
        _, class_name = sample
        grouped[class_name].append(sample)

    train_samples, val_samples, test_samples = [], [], []

    for class_name, class_samples in grouped.items():
        random.shuffle(class_samples)

        n = len(class_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_samples.extend(class_samples[:n_train])
        val_samples.extend(class_samples[n_train:n_train + n_val])
        test_samples.extend(class_samples[n_train + n_val:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    return train_samples, val_samples, test_samples

if __name__ == "__main__":
    samples = build_samples_from_folders()
    train_samples, val_samples, test_samples = stratified_split(samples)
    print(f"Total samples: {len(samples)}")
    print(f"Train: {len(train_samples)}")
    print(f"Val: {len(val_samples)}")
    print(f"Test: {len(test_samples)}")