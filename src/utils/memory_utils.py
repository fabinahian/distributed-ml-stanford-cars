import torch
import gc
import psutil
import os


class GPUMemoryManager:
    def __init__(self, fraction=0.25, verbose=True):
        """
        Initialize GPU memory manager.
        Args:
            fraction (float): Fraction of GPU memory to use (e.g., 0.25 for quarter)
            verbose (bool): Whether to print memory information
        """
        self.fraction = fraction
        self.verbose = verbose
        self.original_memory = None
        self.reserved_memory = None

    def limit_gpu_memory(self):
        """Limit GPU memory usage to specified fraction"""
        if not torch.cuda.is_available():
            if self.verbose:
                print("CUDA is not available. Running on CPU.")
            return

        # Store original memory info
        self.original_memory = torch.cuda.get_device_properties(0).total_memory
        target_memory = int(self.original_memory * self.fraction)

        # Clear cache and garbage collect
        torch.cuda.empty_cache()
        gc.collect()

        try:
            # Try to limit memory using PyTorch's memory allocator
            torch.cuda.set_per_process_memory_limit(target_memory)
            if self.verbose:
                print(
                    f"Successfully limited GPU memory to {target_memory / 1024**3:.2f} GB "
                    f"({self.fraction * 100:.1f}% of total)"
                )
        except (AttributeError, RuntimeError):
            # Fallback: Reserve memory by allocating a large tensor
            remaining_memory = target_memory * 0.95  # Leave some overhead
            self.reserved_memory = torch.empty(
                int(remaining_memory / 4),  # 4 bytes per float32
                dtype=torch.float32,
                device="cuda",
            )
            if self.verbose:
                print(
                    f"Using tensor allocation to limit GPU memory to approximately "
                    f"{remaining_memory / 1024**3:.2f} GB"
                )

    def get_memory_stats(self):
        """
        Get current memory usage statistics.
        Returns:
            dict: Memory statistics for both GPU and system RAM
        """
        stats = {
            "system": {
                "total": psutil.virtual_memory().total / 1024**3,
                "available": psutil.virtual_memory().available / 1024**3,
                "percent": psutil.virtual_memory().percent,
            }
        }

        if torch.cuda.is_available():
            stats["gpu"] = {
                "total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "allocated": torch.cuda.memory_allocated() / 1024**3,
                "reserved": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
            }

        return stats

    def print_memory_stats(self):
        """Print current memory usage statistics"""
        stats = self.get_memory_stats()

        print("\nMemory Usage Statistics:")
        print("-" * 50)

        # System RAM
        print("\nSystem Memory:")
        print(f"Total: {stats['system']['total']:.2f} GB")
        print(f"Available: {stats['system']['available']:.2f} GB")
        print(f"Usage: {stats['system']['percent']}%")

        # GPU Memory
        if "gpu" in stats:
            print("\nGPU Memory:")
            print(f"Total: {stats['gpu']['total']:.2f} GB")
            print(f"Allocated: {stats['gpu']['allocated']:.2f} GB")
            print(f"Reserved: {stats['gpu']['reserved']:.2f} GB")
            print(f"Peak Usage: {stats['gpu']['max_allocated']:.2f} GB")

    def release_memory(self):
        """Release reserved memory if any"""
        if self.reserved_memory is not None:
            del self.reserved_memory
            self.reserved_memory = None

        torch.cuda.empty_cache()
        gc.collect()

        if self.verbose:
            print("Released reserved GPU memory")
            self.print_memory_stats()


def get_gpu_info():
    """
    Get detailed GPU information.
    Returns:
        dict: GPU specifications and capabilities
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    return {
        "name": props.name,
        "total_memory": props.total_memory / 1024**3,
        "major": props.major,
        "minor": props.minor,
        "multi_processor_count": props.multi_processor_count,
        "max_threads_per_block": props.max_threads_per_block,
        "max_threads_per_multiprocessor": props.max_threads_per_multiprocessor,
        "warp_size": props.warp_size,
    }


def print_gpu_info():
    """Print detailed GPU information"""
    info = get_gpu_info()

    if "error" in info:
        print(info["error"])
        return

    print("\nGPU Information:")
    print("-" * 50)
    print(f"Device Name: {info['name']}")
    print(f"Total Memory: {info['total_memory']:.2f} GB")
    print(f"CUDA Capability: {info['major']}.{info['minor']}")
    print(f"Number of SMs: {info['multi_processor_count']}")
    print(f"Max Threads per Block: {info['max_threads_per_block']}")
    print(f"Max Threads per SM: {info['max_threads_per_multiprocessor']}")
    print(f"Warp Size: {info['warp_size']}")


if __name__ == "__main__":
    # Example usage
    print_gpu_info()

    # Initialize memory manager
    memory_mgr = GPUMemoryManager(fraction=0.25, verbose=True)

    # Limit GPU memory
    memory_mgr.limit_gpu_memory()

    # Print initial stats
    memory_mgr.print_memory_stats()

    # Your training code would go here

    # Release memory when done
    memory_mgr.release_memory()
