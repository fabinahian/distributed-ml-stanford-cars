import torch
import gc
import psutil
import os
from typing import Optional
import numpy as np


class MemoryOptimizer:
    def __init__(self, rank: Optional[int] = None, world_size: Optional[int] = None):
        self.rank = rank
        self.world_size = world_size
        self.process = psutil.Process(os.getpid())
        self.last_memory_usage = 0
        self.peak_memory_usage = 0

    def optimize_gpu_memory(self, fraction: float = 0.25):
        """Optimize GPU memory usage"""
        if not torch.cuda.is_available():
            return

        # Empty cache and collect garbage
        torch.cuda.empty_cache()
        gc.collect()

        # Set memory fraction
        if self.world_size is not None:
            fraction = fraction / self.world_size

        # Reserve memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        target_memory = int(total_memory * fraction)

        torch.cuda.set_per_process_memory_limit(target_memory)

    def optimize_dataloader(self, dataloader):
        """Optimize DataLoader settings"""
        dataloader.num_workers = 1
        dataloader.prefetch_factor = 2
        dataloader.persistent_workers = True
        return dataloader

    def get_memory_stats(self):
        """Get current memory usage statistics"""
        stats = {
            "ram": {
                "used": self.process.memory_info().rss / 1024**3,
                "available": psutil.virtual_memory().available / 1024**3,
                "percent": psutil.virtual_memory().percent,
            }
        }

        if torch.cuda.is_available():
            stats["gpu"] = {
                "allocated": torch.cuda.memory_allocated() / 1024**3,
                "reserved": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
            }
            self.peak_memory_usage = max(
                self.peak_memory_usage, stats["gpu"]["allocated"]
            )

        return stats

    def log_memory_stats(self, logger, step: int):
        """Log memory statistics"""
        stats = self.get_memory_stats()

        if self.rank == 0 or self.rank is None:
            logger.info(f"\nStep {step} Memory Usage:")
            logger.info(f"RAM Used: {stats['ram']['used']:.2f}GB")
            logger.info(f"RAM Available: {stats['ram']['available']:.2f}GB")

            if "gpu" in stats:
                logger.info(f"GPU Allocated: {stats['gpu']['allocated']:.2f}GB")
                logger.info(f"GPU Peak: {self.peak_memory_usage:.2f}GB")

    def empty_cache_if_needed(self, iteration: int, freq: int = 10):
        """Empty cache periodically"""
        if iteration % freq == 0:
            torch.cuda.empty_cache()
            gc.collect()

    def check_memory_leak(self, threshold_mb: float = 100):
        """Check for potential memory leaks"""
        current_memory = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        memory_diff = current_memory - self.last_memory_usage

        if memory_diff > threshold_mb:
            return True, memory_diff

        self.last_memory_usage = current_memory
        return False, memory_diff


def optimize_transformer_model(model):
    """Optimize transformer-based models"""
    if hasattr(model, "transformer"):
        # Enable gradient checkpointing
        model.transformer.gradient_checkpointing_enable()

    # Use dynamic padding
    model.train()  # Ensure training mode for dynamic shapes
    return model


def get_optimal_batch_size(model, input_shape, max_memory_gb=4):
    """Estimate optimal batch size based on memory constraints"""
    try:
        torch.cuda.empty_cache()
        gc.collect()

        # Start with batch size of 1
        batch_size = 1
        while True:
            # Try a forward and backward pass
            inputs = torch.randn(batch_size, *input_shape, device="cuda")
            outputs = model(inputs)
            loss = outputs.sum()
            loss.backward()

            # Check memory usage
            memory_used = torch.cuda.memory_allocated() / 1024**3
            if memory_used > max_memory_gb:
                torch.cuda.empty_cache()
                gc.collect()
                return max(1, batch_size - 1)

            batch_size *= 2

    except RuntimeError:  # Out of memory
        torch.cuda.empty_cache()
        gc.collect()
        return max(1, batch_size // 2)
    finally:
        torch.cuda.empty_cache()
        gc.collect()
