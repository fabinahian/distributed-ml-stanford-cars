import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional, List
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import autoaugment, RandomErasing


class CarDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, num_classes: int = 10):
        """
        Custom dataset for car classification.
        Args:
            root_dir (str): Directory with class folders
            transform: Optional transforms to be applied
            num_classes (int): Number of classes to use
        """
        self.root_dir = root_dir
        self.transform = transform

        # Get first num_classes folders
        self.classes = sorted(os.listdir(root_dir))[:num_classes]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.images = []
        self.labels = []
        self.class_counts = {cls: 0 for cls in self.classes}

        # Load dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset paths and labels"""
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.images.append(img_path)
                self.labels.append(class_idx)
                self.class_counts[class_name] += 1

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced dataset handling.
        Returns:
            torch.Tensor: Weights for each class
        """
        total_samples = len(self.images)
        class_weights = torch.zeros(len(self.classes))
        for idx, cls in enumerate(self.classes):
            class_weights[idx] = total_samples / (
                len(self.classes) * self.class_counts[cls]
            )
        return class_weights

    def print_stats(self):
        """Print dataset statistics"""
        print("\nDataset Statistics:")
        print("-" * 50)
        print(f"Total images: {len(self.images)}")
        print("\nClass distribution:")
        for class_name, count in self.class_counts.items():
            print(f"{class_name}: {count} images ({count/len(self.images)*100:.2f}%)")


def create_transforms(train: bool = True, img_size: int = 224) -> transforms.Compose:
    """
    Create transformation pipeline.
    Args:
        train (bool): Whether to create transforms for training
        img_size (int): Target image size
    Returns:
        transforms.Compose: Transformation pipeline
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    img_size, scale=(0.08, 1.0), ratio=(0.75, 1.33)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
                ),
                autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                RandomErasing(
                    p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.14)),  # 256/224 = 1.14
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def create_data_loaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 32,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation and testing.
    Args:
        train_dir (str): Training data directory
        test_dir (str): Test data directory
        batch_size (int): Batch size
        world_size (int, optional): Number of distributed processes
        rank (int, optional): Rank of current process
        num_workers (int): Number of data loading workers
        pin_memory (bool): Whether to pin memory in data loader
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation and test loaders
    """
    # Create transforms
    train_transform = create_transforms(train=True)
    test_transform = create_transforms(train=False)

    # Create datasets
    train_dataset = CarDataset(train_dir, transform=train_transform)
    test_dataset = CarDataset(test_dir, transform=test_transform)

    # Print dataset information
    if rank is None or rank == 0:
        train_dataset.print_stats()
        test_dataset.print_stats()

    # Split train into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size], generator=generator
    )

    # Create samplers if distributed
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        if world_size
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        if world_size
        else None
    )
    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        if world_size
        else None
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data pipeline
    train_dir = "car_data/train"
    test_dir = "car_data/test"

    # Test in non-distributed mode
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dir, test_dir, batch_size=32
    )

    # Print example batch information
    print("\nExample batch information:")
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels}")
        break
