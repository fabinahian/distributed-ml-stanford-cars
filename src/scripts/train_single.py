import os
import torch
import torch.nn as nn
import argparse
from datetime import datetime
import yaml
from pathlib import Path

# Add parent directory to path
import sys

parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.data.data_preprocessing import create_data_loaders
from src.model.model_setup import setup_model
from src.training.trainer import Trainer
from src.utils.memory_utils import GPUMemoryManager


def parse_args():
    parser = argparse.ArgumentParser(description="Single GPU Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/single_node_config.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="data/car_data/train",
        help="path to training data",
    )
    parser.add_argument(
        "--test-dir", type=str, default="data/car_data/test", help="path to test data"
    )
    parser.add_argument("--num-classes", type=int, default=10, help="number of classes")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--num-epochs", type=int, default=5, help="number of epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=0.0001, help="learning rate"
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=1.0,
        help="fraction of GPU memory to use",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=f'single_gpu_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        help="name for the experiment",
    )
    return parser.parse_args()
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(passed_args=None):
    # Use passed args or parse from command line
    if passed_args is None:
        args = parse_args()

        # Load config if exists
        if os.path.exists(args.config):
            config = load_config(args.config)
            # Update args with config values
            for k, v in config.items():
                if not hasattr(args, k):
                    setattr(args, k, v)
    else:
        args = passed_args

    # Setup device and memory management
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        memory_manager = GPUMemoryManager(fraction=args.memory_fraction)
        memory_manager.limit_gpu_memory()
        memory_manager.print_memory_stats()

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dir=args.train_dir, test_dir=args.test_dir, batch_size=args.batch_size
    )

    # Setup model, criterion, optimizer
    model, optimizer, checkpointer = setup_model(num_classes=args.num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=50, verbose=True
    )

    # Use default experiment name if not provided
    if not hasattr(args, "experiment_name"):
        args.experiment_name = f'single_gpu_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        checkpointer=checkpointer,
        scheduler=scheduler,
        experiment_name=args.experiment_name,
    )

    # Print training configuration
    print("\nTraining Configuration:")
    print("-" * 50)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print(f"Device: {device}")
    print("-" * 50)

    # Start training
    trainer.train()

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_accuracy = 100.0 * correct / total
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")

    # Final memory stats
    if torch.cuda.is_available():
        memory_manager.print_memory_stats()


if __name__ == "__main__":
    main()
