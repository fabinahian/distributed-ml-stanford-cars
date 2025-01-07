import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple
import time
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import logging
from ..utils.memory_utils import GPUMemoryManager


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_epochs: int,
        checkpointer: Optional[object] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        experiment_name: str = None,
    ):
        """
        Initialize trainer for single GPU training.
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            num_epochs: Number of epochs to train
            checkpointer: Model checkpointing handler
            scheduler: Learning rate scheduler
            experiment_name: Name for the experiment (for logging)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.checkpointer = checkpointer
        self.scheduler = scheduler

        # Setup experiment name and directories
        self.experiment_name = (
            experiment_name or f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        self.runs_dir = Path("runs") / self.experiment_name
        self.plots_dir = Path("plots") / self.experiment_name
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tensorboard writer
        self.writer = SummaryWriter(self.runs_dir)

        # Setup logging
        self.setup_logging()

        # Initialize metrics storage
        self.metrics = {
            "train_losses": [],
            "train_accuracies": [],
            "val_losses": [],
            "val_accuracies": [],
            "batch_times": [],
            "epoch_times": [],
        }

        # Initialize memory manager
        self.memory_manager = GPUMemoryManager(verbose=True)

    def setup_logging(self):
        """Setup logging configuration"""
        # Check if this is a distributed trainer
        is_distributed = hasattr(self, "rank")
        main_process = not is_distributed or (is_distributed and self.rank == 0)

        if main_process:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler(self.runs_dir / "training.log"),
                    logging.StreamHandler(),
                ],
            )
        else:
            # For distributed training, other ranks only log to file
            log_file = self.runs_dir / f"rank_{self.rank}_training.log"
            logging.basicConfig(
                level=logging.INFO,
                format=f"%(asctime)s - %(levelname)s - [Rank {self.rank}] %(message)s",
                handlers=[logging.FileHandler(log_file)],
            )

        self.logger = logging.getLogger(__name__)

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        Args:
            epoch: Current epoch number
        Returns:
            Tuple of average loss and accuracy
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            batch_start_time = time.time()

            # Move data to device
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Record metrics
            total_loss += loss.item()
            batch_time = time.time() - batch_start_time
            self.metrics["batch_times"].append(batch_time)

            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar("Batch/Loss", loss.item(), global_step)
            self.writer.add_scalar("Batch/Time", batch_time, global_step)

            # Print progress
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Acc: {100.*correct/total:.2f}% "
                    f"Time: {batch_time:.3f}s"
                )

            # Monitor memory usage
            if batch_idx % 50 == 0:
                memory_stats = self.memory_manager.get_memory_stats()
                self.writer.add_scalar(
                    "Memory/GPU_Used", memory_stats["gpu"]["allocated"], global_step
                )

        epoch_time = time.time() - epoch_start_time
        self.metrics["epoch_times"].append(epoch_time)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        self.metrics["train_losses"].append(avg_loss)
        self.metrics["train_accuracies"].append(accuracy)

        return avg_loss, accuracy

    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validate the model.
        Args:
            epoch: Current epoch number
        Returns:
            Tuple of average loss and accuracy
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        self.metrics["val_losses"].append(avg_loss)
        self.metrics["val_accuracies"].append(accuracy)

        return avg_loss, accuracy

    def save_training_plots(self):
        """Generate and save training visualization plots"""
        plt.figure(figsize=(15, 10))

        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics["train_losses"], label="Train Loss")
        plt.plot(self.metrics["val_losses"], label="Val Loss")
        plt.title("Loss Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Accuracy plot
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics["train_accuracies"], label="Train Accuracy")
        plt.plot(self.metrics["val_accuracies"], label="Val Accuracy")
        plt.title("Accuracy Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        # Training time plots
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics["epoch_times"])
        plt.title("Epoch Training Time")
        plt.xlabel("Epoch")
        plt.ylabel("Time (seconds)")

        plt.subplot(2, 2, 4)
        plt.plot(self.metrics["batch_times"])
        plt.title("Batch Processing Time")
        plt.xlabel("Batch")
        plt.ylabel("Time (seconds)")

        plt.tight_layout()
        plt.savefig(self.plots_dir / "training_metrics.png")
        plt.close()

        # Save metrics to JSON
        with open(self.runs_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=4)

    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting training for {self.num_epochs} epochs...")
        best_val_acc = 0.0

        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            # Learning rate scheduling - pass validation accuracy as metric
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_acc)  # Use validation accuracy as metric
                else:
                    self.scheduler.step()

            # Log to tensorboard
            self.writer.add_scalar("Epoch/Train_Loss", train_loss, epoch)
            self.writer.add_scalar("Epoch/Train_Accuracy", train_acc, epoch)
            self.writer.add_scalar("Epoch/Val_Loss", val_loss, epoch)
            self.writer.add_scalar("Epoch/Val_Accuracy", val_acc, epoch)
            self.writer.add_scalar("Epoch/Time", self.metrics["epoch_times"][-1], epoch)

            # Save checkpoint
            if self.checkpointer is not None:
                is_best = val_acc > best_val_acc
                best_val_acc = max(val_acc, best_val_acc)
                self.checkpointer.save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    {
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    is_best,
                )

            # Log epoch summary
            self.logger.info(
                f"\nEpoch {epoch} Summary:\n"
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n"
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n"
                f"Epoch Time: {self.metrics['epoch_times'][-1]:.2f}s\n"
            )

        # Save final plots and metrics
        self.save_training_plots()
        self.writer.close()
        self.logger.info("Training completed!")


if __name__ == "__main__":
    # Example usage
    from ..model.model_setup import setup_model
    from ..data.data_preprocessing import create_data_loaders

    # Setup
    model, optimizer, checkpointer = setup_model()
    train_loader, val_loader, _ = create_data_loaders("data/train", "data/test")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_epochs=5,
        checkpointer=checkpointer,
    )

    trainer.train()
