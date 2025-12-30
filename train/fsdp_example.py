"""
Minimal FSDP2 Training Example with Profiler

This script demonstrates:
1. A simple 3-layer MLP model
2. FSDP2 training using the newer fully_shard API on 2 GPUs (if available) or 2 CPUs
3. Dummy dataloader
4. PyTorch profiler for trace generation

The code uses the newer FSDP2 API (torch.distributed.fsdp.fully_shard) instead of
the older FullyShardedDataParallel wrapper for a more Pythonic and compositional approach.

Usage:
    # For GPU training (if available):
    torchrun --nproc_per_node=2 train/fsdp_example.py
    
    # For CPU training:
    torchrun --nproc_per_node=2 train/fsdp_example.py --cpu
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import profile, record_function, ProfilerActivity


class SimpleMLP(nn.Module):
    """Simple 3-layer MLP for demonstration"""
    
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=512):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class DummyDataset(Dataset):
    """Dummy dataset for testing"""
    
    def __init__(self, num_samples=1000, input_dim=1024, output_dim=512):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random input and target
        x = torch.randn(self.input_dim)
        y = torch.randn(self.output_dim)
        return x, y


def setup_distributed(use_cpu=False):
    """Initialize distributed training"""
    # Force CPU if CUDA is not available (e.g., on macOS with MPS)
    if not torch.cuda.is_available():
        use_cpu = True
    
    # Use gloo backend for CPU, nccl for GPU
    backend = "gloo" if use_cpu else "nccl"
    init_process_group(backend=backend)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    
    if use_cpu:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    
    return rank, local_rank, world_size, device, use_cpu


def cleanup_distributed():
    """Cleanup distributed training"""
    destroy_process_group()


def train_step(model, dataloader, optimizer, criterion, device, rank):
    """Single training step"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if rank == 0 and batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main(args):
    # Setup distributed training
    rank, local_rank, world_size, device, use_cpu = setup_distributed(args.cpu)
    
    if rank == 0:
        print(f"Training on {world_size} {'CPUs' if use_cpu else 'GPUs'}")
        print(f"Device: {device}")
        if use_cpu and not args.cpu:
            print("Note: CUDA not available, automatically using CPU")
    
    # Create device mesh explicitly to avoid MPS detection issues on macOS
    # This forces the use of CPU or CUDA depending on availability
    device_type = "cpu" if use_cpu else "cuda"
    mesh = init_device_mesh(device_type, (world_size,))
    
    if rank == 0:
        print(f"Device mesh created with device_type: {device_type}")
    
    # Create model on the appropriate device
    # For CPU: keep on CPU before FSDP wrapping
    # For GPU: FSDP will handle device placement
    if use_cpu:
        model = SimpleMLP(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim
        )
    else:
        model = SimpleMLP(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim
        ).to(device)
    
    # Apply FSDP2 using fully_shard (newer API) with explicit mesh
    # Shard each layer individually for fine-grained control
    model.layer1 = fully_shard(model.layer1, mesh=mesh)
    model.layer2 = fully_shard(model.layer2, mesh=mesh)
    model.layer3 = fully_shard(model.layer3, mesh=mesh)
    
    # Shard the entire model
    fsdp_model = fully_shard(model, mesh=mesh)
    
    if rank == 0:
        print(f"Model wrapped with FSDP2 (fully_shard API)")
        print(f"Model parameters: {sum(p.numel() for p in fsdp_model.parameters()):,}")
    
    # Create dummy dataset and dataloader
    dataset = DummyDataset(
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        output_dim=args.output_dim
    )
    
    # Use DistributedSampler for data parallelism
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=not use_cpu
    )
    
    # Setup optimizer and loss
    optimizer = optim.AdamW(fsdp_model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    if rank == 0:
        print(f"\nStarting training for {args.epochs} epochs...")
        print(f"Batch size: {args.batch_size}, Learning rate: {args.lr}")
        print(f"Dataset size: {len(dataset)}, Batches per epoch: {len(dataloader)}\n")
    
    # Training loop with profiler
    activities = [ProfilerActivity.CPU]
    if not use_cpu:
        activities.append(ProfilerActivity.CUDA)
    
    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./trace_logs/rank_{rank}"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)  # Important for proper shuffling
            
            with record_function(f"epoch_{epoch}"):
                avg_loss = train_step(
                    fsdp_model,
                    dataloader,
                    optimizer,
                    criterion,
                    device,
                    rank
                )
            
            if rank == 0:
                print(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
            
            prof.step()  # Signal the profiler to move to the next step
    
    if rank == 0:
        print(f"\nTraining completed!")
        print(f"Profiler traces saved to: ./trace_logs/rank_{rank}")
        print(f"View traces with: tensorboard --logdir=./trace_logs")
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSDP2 Training Example")
    
    # Model parameters
    parser.add_argument("--input-dim", type=int, default=1024, help="Input dimension")
    parser.add_argument("--hidden-dim", type=int, default=2048, help="Hidden dimension")
    parser.add_argument("--output-dim", type=int, default=512, help="Output dimension")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples")
    
    # Device parameters
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    
    args = parser.parse_args()
    
    main(args)

