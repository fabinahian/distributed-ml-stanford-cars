import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Dict, Any, Tuple
import json
from pathlib import Path


class CarClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        """
        Car classifier model based on ResNet18.
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
        """
        super(CarClassifier, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)

        # Modify the final layer for our number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_parameter_groups(
        self,
    ) -> Tuple[Dict[str, nn.Parameter], Dict[str, nn.Parameter]]:
        """
        Get separate parameter groups for backbone and head.
        Returns:
            Tuple[Dict]: Backbone and head parameters
        """
        backbone_params = {
            k: v for k, v in self.model.named_parameters() if not k.startswith("fc.")
        }
        head_params = {
            k: v for k, v in self.model.named_parameters() if k.startswith("fc.")
        }
        return backbone_params, head_params


class ModelCheckpointing:
    def __init__(
        self, save_dir: str = "checkpoints", filename_prefix: str = "checkpoint"
    ):
        """
        Handle model checkpointing for distributed training.
        Args:
            save_dir (str): Directory to save checkpoints
            filename_prefix (str): Prefix for checkpoint filenames
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filename_prefix = filename_prefix
        self.best_val_acc = 0.0
        self.metrics_file = self.save_dir / "training_metrics.json"
        self.metrics_history = self._load_metrics_history()

    def _load_metrics_history(self) -> Dict:
        """Load existing metrics history if it exists"""
        if self.metrics_file.exists():
            with open(self.metrics_file, "r") as f:
                return json.load(f)
        return {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        rank: int = 0,
    ) -> None:
        """
        Save model checkpoint and metrics.
        Args:
            model (nn.Module): Model to save
            optimizer (torch.optim.Optimizer): Optimizer to save
            epoch (int): Current epoch
            metrics (Dict): Current metrics
            is_best (bool): Whether this is the best model so far
            rank (int): Process rank in distributed training
        """
        # Only save checkpoints on rank 0
        if rank != 0:
            return

        # Update metrics history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

        # Save metrics history
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=4)

        # Prepare checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": (
                model.module.state_dict()
                if isinstance(model, DDP)
                else model.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        # Save latest checkpoint
        latest_path = self.save_dir / f"{self.filename_prefix}_latest.pt"
        torch.save(checkpoint, latest_path)

        # Save numbered checkpoint every N epochs
        if epoch % 5 == 0:
            numbered_path = self.save_dir / f"{self.filename_prefix}_epoch_{epoch}.pt"
            torch.save(checkpoint, numbered_path)

        # Save best checkpoint if needed
        if is_best:
            best_path = self.save_dir / f"{self.filename_prefix}_best.pt"
            torch.save(checkpoint, best_path)
            self.best_val_acc = metrics.get("val_acc", 0.0)

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device("cpu"),
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[int, Dict[str, float]]:
        """
        Load model checkpoint.
        Args:
            model (nn.Module): Model to load weights into
            optimizer (torch.optim.Optimizer, optional): Optimizer to load state into
            device (torch.device): Device to load checkpoint to
            checkpoint_path (str, optional): Specific checkpoint to load
        Returns:
            Tuple[int, Dict]: Epoch number and metrics
        """
        if checkpoint_path is None:
            checkpoint_path = self.save_dir / f"{self.filename_prefix}_latest.pt"

        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}")
            return 0, {}

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint["epoch"], checkpoint.get("metrics", {})


def setup_model(
    rank: Optional[int] = None, world_size: Optional[int] = None, num_classes: int = 10
) -> Tuple[nn.Module, torch.optim.Optimizer, ModelCheckpointing]:
    """
    Setup model, optimizer and checkpointing for training.
    Args:
        rank (int, optional): Process rank for distributed training
        world_size (int, optional): Total number of processes
        num_classes (int): Number of classes for classification
    Returns:
        Tuple: Model, optimizer and checkpointing handler
    """
    # Create model
    model = CarClassifier(num_classes=num_classes)

    # Setup device - always use cuda:0 for single GPU simulation
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model = model.to(device)
    else:
        device = torch.device("cpu")
        model = model.to(device)

    # Wrap model with DDP if in distributed mode
    if rank is not None and world_size is not None:
        model = DDP(model, device_ids=[0], output_device=0)  # Always use first GPU

    # Get separate parameter groups for different learning rates
    backbone_params, head_params = (
        model.module.get_parameter_groups()
        if isinstance(model, DDP)
        else model.get_parameter_groups()
    )

    # Initialize optimizer with parameter groups
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params.values(), "lr": 1e-4},  # Lower LR for backbone
            {"params": head_params.values(), "lr": 1e-3},  # Higher LR for head
        ],
        weight_decay=0.01,
    )

    # Initialize checkpointing
    checkpointer = ModelCheckpointing()

    return model, optimizer, checkpointer


def print_model_info(model: nn.Module) -> None:
    """Print model information"""
    print("\nModel Information:")
    print("-" * 50)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    # Test model setup
    model, optimizer, checkpointer = setup_model()

    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    if torch.cuda.is_available():
        x = x.cuda()
    output = model(x)

    print_model_info(model)
    print(f"\nOutput shape: {output.shape}")
