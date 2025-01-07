import os
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
from pathlib import Path
from datetime import datetime
import yaml

# Add parent directory to path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.data_preprocessing import create_data_loaders
from src.model.model_setup import setup_model
from src.training.distributed_trainer import DistributedTrainer
from src.utils.memory_utils import GPUMemoryManager
from src.utils.distributed_utils import setup_distributed_simulator


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/distributed_config.yaml",
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
    parser.add_argument("--batch-size", type=int, default=16, help="batch size per GPU")
    parser.add_argument("--num-epochs", type=int, default=5, help="number of epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=0.0001, help="learning rate"
    )
    parser.add_argument(
        "--num-nodes", type=int, default=4, help="number of nodes to simulate"
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.23,
        help="fraction of GPU memory per node",
    )
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def setup_environment(rank, world_size, memory_fraction):
    """Setup distributed environment and GPU memory management"""
    # Initialize distributed environment
    dist_manager = setup_distributed_simulator(rank, world_size)

    # Setup GPU memory management
    memory_manager = GPUMemoryManager(fraction=memory_fraction)
    memory_manager.limit_gpu_memory()

    if rank == 0:
        memory_manager.print_memory_stats()
        print(f"\nProcess {rank}: Using {memory_fraction:.1%} of GPU memory")

    return dist_manager, memory_manager


def run_worker(rank, world_size, args):
    """
    Run training on a single worker process.
    Args:
        rank: Process rank
        world_size: Total number of processes
        args: Training arguments
    """
    try:
        # Calculate memory fraction per node
        memory_fraction = args.memory_fraction / world_size

        # Setup environment
        dist_manager, memory_manager = setup_environment(
            rank, world_size, memory_fraction
        )

        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            batch_size=args.batch_size,
            world_size=world_size,
            rank=rank,
            num_workers=1,  # Reduced for memory optimization
            pin_memory=True,
        )

        # Setup model and optimizer
        model, optimizer, checkpointer = setup_model(
            rank=rank, world_size=world_size, num_classes=args.num_classes
        )

        # Setup learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=50, verbose=(rank == 0)
        )

        # Create experiment name
        experiment_name = (
            f'distributed_{world_size}nodes_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )

        # Initialize trainer
        trainer = DistributedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            device=torch.device("cuda:0"),  # Always use first GPU
            num_epochs=args.num_epochs,
            rank=rank,
            world_size=world_size,
            checkpointer=checkpointer,
            scheduler=scheduler,
            experiment_name=experiment_name,
        )

        # Start training
        if rank == 0:
            print(f"\nStarting distributed training with {world_size} nodes")
            print(f"Batch size per node: {args.batch_size}")
            print(f"Effective batch size: {args.batch_size * world_size}")
            print(f"Number of epochs: {args.num_epochs}")

        trainer.train()

        # Cleanup
        dist_manager.cleanup()

        if rank == 0:
            print("\nTraining completed successfully!")
            memory_manager.print_memory_stats()

    except Exception as e:
        print(f"Error in worker {rank}: {str(e)}")
        raise e


def main():
    """Main function for distributed training"""
    args = parse_args()

    # Load and merge configuration
    config = load_config(args.config)
    for k, v in config.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # Validate GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distributed training")

    # Print configuration
    print("\nDistributed Training Configuration:")
    print("-" * 50)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("-" * 50)

    try:
        import torch.multiprocessing as mp

        mp.spawn(
            run_worker, args=(args.num_nodes, args), nprocs=args.num_nodes, join=True
        )
    except Exception as e:
        print(f"\nError during distributed training: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
