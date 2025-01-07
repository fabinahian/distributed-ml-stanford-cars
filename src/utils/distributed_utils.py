import os
import torch
import torch.distributed as dist
import socket
from typing import Optional, Tuple


class DistributedManager:
    def __init__(
        self,
        backend: str = None,  # Changed to None to auto-select
        init_method: str = "env://",
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        master_addr: str = "localhost",
        master_port: str = "12355",
    ):
        """
        Initialize distributed training manager.
        Args:
            backend (str): PyTorch distributed backend ('gloo' for CPU/Windows, 'nccl' for Linux GPU)
            init_method (str): Initialization method for process group
            world_size (int, optional): Total number of processes
            rank (int, optional): Global rank of current process
            local_rank (int, optional): Local rank of current process
            master_addr (str): Address of master node
            master_port (str): Port for distributed communication
        """
        # Auto-select backend based on platform and CUDA availability
        if backend is None:
            if torch.cuda.is_available() and os.name != "nt":  # Linux with CUDA
                self.backend = "nccl"
            else:  # Windows or CPU-only
                self.backend = "gloo"
        else:
            self.backend = backend

        self.init_method = init_method
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.master_addr = master_addr
        self.master_port = master_port

    def setup(self) -> None:
        """
        Setup distributed environment.
        """
        # Set environment variables
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = self.master_port

        if self.world_size is not None:
            os.environ["WORLD_SIZE"] = str(self.world_size)
        if self.rank is not None:
            os.environ["RANK"] = str(self.rank)
        if self.local_rank is not None:
            os.environ["LOCAL_RANK"] = str(self.local_rank)

        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                init_method=self.init_method,
                world_size=self.world_size,
                rank=self.rank,
            )

        # Set device for this process
        if torch.cuda.is_available():
            if self.local_rank is not None:
                torch.cuda.set_device(self.local_rank)
            else:
                torch.cuda.set_device(self.rank % torch.cuda.device_count())

    def cleanup(self) -> None:
        """
        Cleanup distributed environment.
        """
        if dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    def get_dist_info() -> Tuple[int, int]:
        """
        Get distributed training information.
        Returns:
            Tuple[int, int]: rank and world_size
        """
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        return rank, world_size

    @staticmethod
    def is_main_process() -> bool:
        """
        Check if current process is the main process (rank 0).
        Returns:
            bool: True if main process
        """
        return not dist.is_initialized() or dist.get_rank() == 0

    @staticmethod
    def synchronize() -> None:
        """
        Synchronize all processes in distributed training.
        """
        if not dist.is_initialized():
            return
        if dist.get_world_size() == 1:
            return
        dist.barrier()


def setup_distributed_simulator(
    node_rank: int, num_nodes: int = 4
) -> DistributedManager:
    """
    Setup distributed training simulation on a single GPU.
    Args:
        node_rank (int): Current node rank (0 to num_nodes-1)
        num_nodes (int): Total number of nodes to simulate
    Returns:
        DistributedManager: Initialized distributed manager
    """
    assert 0 <= node_rank < num_nodes, f"Invalid node rank: {node_rank}"

    # Create manager without specifying backend (will auto-select)
    manager = DistributedManager(
        world_size=num_nodes,
        rank=node_rank,
        local_rank=0,  # Since we're simulating on a single GPU
    )

    print(f"Initializing process group with backend: {manager.backend}")
    manager.setup()

    return manager


class DistributedMetricsAggregator:
    """Helper class for aggregating metrics across distributed processes"""

    @staticmethod
    def all_reduce_average(tensor: torch.Tensor) -> torch.Tensor:
        """
        Average the tensor across all processes.
        Args:
            tensor (torch.Tensor): Tensor to be averaged
        Returns:
            torch.Tensor: Averaged tensor
        """
        if not dist.is_initialized():
            return tensor

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor / dist.get_world_size()

    @staticmethod
    def all_gather(tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather tensor from all processes.
        Args:
            tensor (torch.Tensor): Tensor to be gathered
        Returns:
            torch.Tensor: Gathered tensor
        """
        if not dist.is_initialized():
            return tensor

        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)


def print_distributed_info():
    """Print information about the distributed setup"""
    print("\nDistributed Training Information:")
    print("-" * 50)
    print(
        f"Backend: {dist.get_backend() if dist.is_initialized() else 'Not initialized'}"
    )
    print(f"World Size: {dist.get_world_size() if dist.is_initialized() else 1}")
    print(f"Rank: {dist.get_rank() if dist.is_initialized() else 0}")
    print(f"Local Rank: {os.environ.get('LOCAL_RANK', '0')}")
    print(f"Master Address: {os.environ.get('MASTER_ADDR', 'Not set')}")
    print(f"Master Port: {os.environ.get('MASTER_PORT', 'Not set')}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.current_device()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name()}")
    print(f"Machine: {socket.gethostname()}")


if __name__ == "__main__":
    # Example usage
    manager = setup_distributed_simulator(node_rank=0, num_nodes=4)
    print_distributed_info()

    # Your distributed training code would go here

    manager.cleanup()
