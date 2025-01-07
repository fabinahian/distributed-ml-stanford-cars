import os
import argparse
import yaml
import torch
import torch.multiprocessing as mp
from pathlib import Path
from datetime import datetime
import json

# Fix imports
import sys

parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from .train_single import main as run_single
from .train_distributed import run_worker


class ExperimentLauncher:
    def __init__(self):
        self.args = self.parse_arguments()
        self.config = self.load_config()
        self.experiment_dir = self.setup_experiment_directory()

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Training Launcher")
        parser.add_argument(
            "--mode",
            type=str,
            choices=["single", "distributed"],
            default="single",
            help="training mode (single GPU or distributed)",
        )
        parser.add_argument(
            "--config", type=str, default=None, help="path to config file"
        )
        parser.add_argument(
            "--experiment-name", type=str, default=None, help="name for the experiment"
        )
        parser.add_argument(
            "--num-nodes",
            type=int,
            default=4,
            help="number of nodes for distributed training",
        )
        parser.add_argument(
            "--memory-fraction",
            type=float,
            default=None,
            help="fraction of GPU memory to use (for single GPU mode)",
        )
        parser.add_argument(
            "--train-dir",
            type=str,
            default="data/car_data/train",
            help="path to training data",
        )
        parser.add_argument(
            "--test-dir",
            type=str,
            default="data/car_data/test",
            help="path to test data",
        )
        parser.add_argument(
            "--batch-size", type=int, default=None, help="batch size per GPU"
        )
        parser.add_argument(
            "--num-epochs", type=int, default=None, help="number of epochs"
        )
        parser.add_argument(
            "--learning-rate", type=float, default=None, help="learning rate"
        )

        return parser.parse_args()

    def load_config(self):
        """Load and merge configuration from file and command line arguments"""
        config = {}

        # Default config file based on mode
        default_config = f"configs/{self.args.mode}_node_config.yaml"
        config_file = self.args.config or default_config

        # Load config file if exists
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = yaml.safe_load(f) or {}

        # Override config with command line arguments
        for arg in vars(self.args):
            value = getattr(self.args, arg)
            if value is not None:
                config[arg] = value

        # Ensure mode is in config
        config["mode"] = self.args.mode

        # Set timestamp for default experiment name
        config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        return config

    def setup_experiment_directory(self):
        """Create experiment directory and save configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Generate experiment name if not provided
        if self.config.get("experiment_name") is None:
            self.config["experiment_name"] = f"{self.args.mode}_{timestamp}"

        experiment_dir = Path("experiments") / self.config["experiment_name"]
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_path = experiment_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        return experiment_dir

    def print_config(self):
        """Print experiment configuration"""
        print("\nExperiment Configuration:")
        print("-" * 50)
        print(f"Mode: {self.args.mode}")
        print(f"Experiment Directory: {self.experiment_dir}")
        print("\nParameters:")
        for key, value in self.config.items():
            print(f"{key}: {value}")
        print("-" * 50)

    def run_experiment(self):
        """Run the experiment based on mode"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for training")

        self.print_config()

        if self.args.mode == "single":
            self.run_single_gpu()
        else:
            self.run_distributed()

    def run_single_gpu(self):
        """Run single GPU training"""
        print("\nStarting Single GPU Training...")

        # Ensure experiment name is set
        if "experiment_name" not in self.config:
            self.config["experiment_name"] = (
                f'single_gpu_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )

        # Update args with experiment directory
        self.config["experiment_dir"] = str(self.experiment_dir)

        # Convert config to namespace for compatibility
        args = argparse.Namespace(**self.config)

        # Run single GPU training with args
        run_single(passed_args=args)

    def run_distributed(self):
        """Run distributed training"""
        print(
            f"\nStarting Distributed Training with {self.config['num_nodes']} nodes..."
        )

        # Ensure all required configs are present
        default_config = {
            "batch_size": 8,
            "num_epochs": 5,
            "learning_rate": 0.0001,
            "num_classes": 10,
            "train_dir": "data/car_data/train",
            "test_dir": "data/car_data/test",
            "memory_fraction": 0.25,
            "num_workers": 4,
            "pin_memory": True,
        }

        # Update config with defaults for missing values
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value

        # Update config with experiment directory
        self.config["experiment_dir"] = str(self.experiment_dir)

        # Convert config to namespace for compatibility
        args = argparse.Namespace(**self.config)

        # Start multiple processes for distributed training
        mp.spawn(
            run_worker,
            args=(self.config["num_nodes"], args),
            nprocs=self.config["num_nodes"],
            join=True,
        )

    def save_results(self, results):
        """Save experiment results"""
        results_path = self.experiment_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)


def main():
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)

    # Create and run launcher
    launcher = ExperimentLauncher()
    try:
        launcher.run_experiment()
    except Exception as e:
        print(f"\nError during experiment: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
