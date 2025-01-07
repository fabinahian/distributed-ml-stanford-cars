def train(self):
    """Main training loop for distributed training"""
    self.logger.info(f"Starting distributed training on rank {self.rank}")
    self.logger.info(f"World size: {self.world_size}")

    best_val_acc = 0.0

    for epoch in range(self.num_epochs):
        train_loss, train_acc = self.train_epoch(epoch)
        val_loss, val_acc = self.validate(epoch)

        # Learning rate scheduling
        if self.scheduler is not None:
            self.scheduler.step()

        # Logging and checkpointing (rank 0 only)
        if self.rank == 0:
            if self.writer:
                self.writer.add_scalar("Epoch/Train_Loss", train_loss, epoch)
                self.writer.add_scalar("Epoch/Train_Accuracy", train_acc, epoch)
                self.writer.add_scalar("Epoch/Val_Loss", val_loss, epoch)
                self.writer.add_scalar("Epoch/Val_Accuracy", val_acc, epoch)
                self.writer.add_scalar(
                    "Epoch/Time", self.metrics["epoch_times"][-1], epoch
                )
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("Learning_Rate", current_lr, epoch)

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
                        "learning_rate": current_lr,
                    },
                    is_best,
                    rank=self.rank,
                )

            self.logger.info(
                f"\nEpoch {epoch} Summary:\n"
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n"
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n"
                f"Learning Rate: {current_lr:.6f}\n"
                f"Best Val Acc: {best_val_acc:.2f}%\n"
                f"Epoch Time: {self.metrics['epoch_times'][-1]:.2f}s\n"
            )

        # Make sure all processes are synchronized
        if self.world_size > 1:
            dist.barrier()

    # Cleanup
    if self.rank == 0:
        self.save_training_plots()
        if self.writer:
            self.writer.close()
        self.logger.info("Distributed training completed!")


import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple
import time
from pathlib import Path
from datetime import datetime
import logging
from .trainer import Trainer
from ..utils.memory_utils import GPUMemoryManager
from ..utils.distributed_utils import DistributedMetricsAggregator
import matplotlib.pyplot as plt


class DistributedTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_epochs: int,
        rank: int,
        world_size: int,
        checkpointer: Optional[object] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        experiment_name: str = None,
    ):
        """
        Initialize distributed trainer.
        Args:
            model: Neural network model (should be wrapped in DistributedDataParallel)
            train_loader: Training data loader with DistributedSampler
            val_loader: Validation data loader with DistributedSampler
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            num_epochs: Number of epochs to train
            rank: Process rank
            world_size: Total number of processes
            checkpointer: Model checkpointing handler
            scheduler: Learning rate scheduler
            experiment_name: Name for the experiment
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            checkpointer=checkpointer,
            scheduler=scheduler,
            experiment_name=experiment_name,
        )

        self.rank = rank
        self.world_size = world_size
        self.metrics_aggregator = DistributedMetricsAggregator()

        # Only create writer on rank 0
        if self.rank != 0:
            self.writer = None

        # Initialize memory tracking for this process
        self.process_memory = []

    # Removed setup_logging as it's now handled in the base class

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch in distributed setting.
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

        # Set epoch for distributed sampler
        self.train_loader.sampler.set_epoch(epoch)

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            batch_start_time = time.time()

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            # Calculate local accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Record metrics
            total_loss += loss.item()
            batch_time = time.time() - batch_start_time
            self.metrics["batch_times"].append(batch_time)

            # Track memory usage
            if batch_idx % 50 == 0:
                memory_stats = self.memory_manager.get_memory_stats()
                self.process_memory.append(memory_stats["gpu"]["allocated"])

            # Log to tensorboard (rank 0 only)
            if self.rank == 0 and self.writer:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar(
                    f"Batch/Loss_Rank{self.rank}", loss.item(), global_step
                )
                self.writer.add_scalar(
                    f"Batch/Time_Rank{self.rank}", batch_time, global_step
                )

            # Print progress (rank 0 only)
            if self.rank == 0 and batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Acc: {100.*correct/total:.2f}% "
                    f"Time: {batch_time:.3f}s"
                )

        epoch_time = time.time() - epoch_start_time
        self.metrics["epoch_times"].append(epoch_time)

        # Aggregate metrics across processes
        avg_loss = self.aggregate_metric(total_loss / len(self.train_loader))
        accuracy = self.aggregate_metric(100.0 * correct / total)

        self.metrics["train_losses"].append(avg_loss)
        self.metrics["train_accuracies"].append(accuracy)

        return avg_loss, accuracy

    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validate the model in distributed setting.
        Args:
            epoch: Current epoch number
        Returns:
            Tuple of average loss and accuracy
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        # Set epoch for distributed sampler
        self.val_loader.sampler.set_epoch(epoch)

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Aggregate metrics across processes
        avg_loss = self.aggregate_metric(total_loss / len(self.val_loader))
        accuracy = self.aggregate_metric(100.0 * correct / total)

        self.metrics["val_losses"].append(avg_loss)
        self.metrics["val_accuracies"].append(accuracy)

        return avg_loss, accuracy

    def aggregate_metric(self, metric: float) -> float:
        """
        Aggregate metric across all processes.
        Args:
            metric: Local metric value
        Returns:
            float: Aggregated metric value
        """
        metric_tensor = torch.tensor(metric).to(self.device)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        return metric_tensor.item() / self.world_size

    def save_training_plots(self):
        """Generate and save training visualization plots (rank 0 only)"""
        if self.rank != 0:
            return
        super().save_training_plots()

        # Add distributed training specific plots
        if self.process_memory:
            plt.figure(figsize=(10, 5))
            plt.plot(self.process_memory)
            plt.title(f"GPU Memory Usage - Rank {self.rank}")
            plt.xlabel("Steps (x50)")
            plt.ylabel("Memory (GB)")
            plt.savefig(self.plots_dir / f"memory_usage_rank_{self.rank}.png")
            plt.close()

    # def train(self):
    #     """Main training loop for distributed training"""
    #     self.logger.info(f"Starting distributed training on rank {self.rank}")
    #     self.logger.info(f"World size: {self.world_size}")

    #     best_val_acc = 0.0

    #     for epoch in range(self.num_epochs):
    #         train_loss, train_acc = self.train_epoch(epoch)
    #         val_loss, val_acc = self.validate(epoch)

    #         if self.scheduler is not None:
    #             self.scheduler.step()

    #         # Logging and checkpointing (rank 0 only)
    #         if self.rank == 0:
    #             if self.writer:
    #                 self.writer.add_scalar("Epoch/Train_Loss", train_loss, epoch)
    #                 self.writer.add_scalar("Epoch/Train_Accuracy", train_acc, epoch)
    #                 self.writer.add_scalar("Epoch/Val_Loss", val_loss, epoch)
    #                 self.writer.add_scalar("Epoch/Val_Accuracy", val_acc, epoch)
    #                 self.writer.add_scalar(
    #                     "Epoch/Time", self.metrics["epoch_times"][-1], epoch
    #                 )

    #             if self.checkpointer is not None:
    #                 is_best = val_acc > best_val_acc
    #                 best_val_acc = max(val_acc, best_val_acc)
    #                 self.checkpointer.save_checkpoint(
    #                     self.model,
    #                     self.optimizer,
    #                     epoch,
    #                     {
    #                         "train_loss": train_loss,
    #                         "train_acc": train_acc,
    #                         "val_loss": val_loss,
    #                         "val_acc": val_acc,
    #                     },
    #                     is_best,
    #                     rank=self.rank,
    #                 )

    #             self.logger.info(
    #                 f"\nEpoch {epoch} Summary:\n"
    #                 f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n"
    #                 f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n"
    #                 f"Epoch Time: {self.metrics['epoch_times'][-1]:.2f}s\n"
    #             )

    #     # Cleanup
    #     if self.rank == 0:
    #         self.save_training_plots()
    #         if self.writer:
    #             self.writer.close()
    #         self.logger.info("Distributed training completed!")

    def train(self):
        """Main training loop for distributed training"""
        self.logger.info(f"Starting distributed training on rank {self.rank}")
        self.logger.info(f"World size: {self.world_size}")

        best_val_acc = 0.0

        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            # Learning rate scheduling
            if self.scheduler is not None:
                # If the scheduler requires a metric, pass it explicitly
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)  # Assuming val_loss is the metric
                else:
                    self.scheduler.step()

            # Logging and checkpointing (rank 0 only)
            if self.rank == 0:
                if self.writer:
                    self.writer.add_scalar("Epoch/Train_Loss", train_loss, epoch)
                    self.writer.add_scalar("Epoch/Train_Accuracy", train_acc, epoch)
                    self.writer.add_scalar("Epoch/Val_Loss", val_loss, epoch)
                    self.writer.add_scalar("Epoch/Val_Accuracy", val_acc, epoch)
                    self.writer.add_scalar(
                        "Epoch/Time", self.metrics["epoch_times"][-1], epoch
                    )
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("Learning_Rate", current_lr, epoch)

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
                            "learning_rate": current_lr,
                        },
                        is_best,
                        rank=self.rank,
                    )

                self.logger.info(
                    f"\nEpoch {epoch} Summary:\n"
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n"
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n"
                    f"Learning Rate: {current_lr:.6f}\n"
                    f"Best Val Acc: {best_val_acc:.2f}%\n"
                    f"Epoch Time: {self.metrics['epoch_times'][-1]:.2f}s\n"
                )

            # Make sure all processes are synchronized
            if self.world_size > 1:
                dist.barrier()

        # Cleanup
        if self.rank == 0:
            self.save_training_plots()
            if self.writer:
                self.writer.close()
            self.logger.info("Distributed training completed!")


if __name__ == "__main__":
    # Example usage
    from ..model.model_setup import setup_model
    from ..data.data_preprocessing import create_data_loaders
    from ..utils.distributed_utils import setup_distributed_simulator

    # Setup distributed environment
    rank = 0
    world_size = 4
    dist_manager = setup_distributed_simulator(rank, world_size)

    # Setup model and data
    model, optimizer, checkpointer = setup_model(rank=rank, world_size=world_size)
    train_loader, val_loader, _ = create_data_loaders(
        "data/train", "data/test", world_size=world_size, rank=rank
    )

    trainer = DistributedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device(rank),
        num_epochs=5,
        rank=rank,
        world_size=world_size,
        checkpointer=checkpointer,
    )

    trainer.train()

    # Cleanup
    dist_manager.cleanup()
