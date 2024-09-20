import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,8"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from model_UNet_2D_se_feat128 import UNet2D, CombinedLoss
from train_UNet_2D_se_feat128 import BrainMRI2DDataset, visualize_batch


def get_free_gpu_memory(device):
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    return torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)


def train(args):
    # Set up CUDA devices
    if torch.cuda.is_available():
        all_devices = list(range(torch.cuda.device_count()))

        # Check free memory on each GPU and select those with sufficient memory
        free_memory_threshold = 1 * 1024 * 1024 * 1024  # 4 GB in bytes
        device_ids = [device for device in all_devices if get_free_gpu_memory(device) >= free_memory_threshold]

        if not device_ids:
            print("No GPUs with sufficient memory available, using CPU")
            device_ids = None
        else:
            print(f"Using {len(device_ids)} GPUs: {device_ids}")
    else:
        print("No GPUs available, using CPU")
        device_ids = None

    # Create dataset and dataloader
    dataset = BrainMRI2DDataset(args.train_root_dir, args.slice_range)

    # Start with a smaller batch size
    effective_batch_size = args.batch_size
    batch_size = max(1, effective_batch_size // (len(device_ids) if device_ids else 1))
    accumulation_steps = effective_batch_size // batch_size

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Create the model
    model = UNet2D(in_channels=3, out_channels=1, init_features=128)

    if device_ids:
        model = DataParallel(model, device_ids=device_ids)
        model = model.cuda(device_ids[0])

    # Define loss function and optimizer
    criterion = CombinedLoss()
    # if device_ids:
    #    criterion = criterion.cuda(device_ids[0])

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        model.train()
        for batch_idx, (inputs, targets, _) in enumerate(dataloader):
            try:
                if device_ids:
                    inputs, targets = inputs.cuda(), targets.cuda()

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, targets)
                loss = loss / accumulation_steps  # Normalize the loss

                # Backward pass
                loss.backward()

                # Optimize every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if batch_idx % 10 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{args.num_epochs}], Step [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item() * accumulation_steps:.4f}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("WARNING: ran out of memory, attempting to recover")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    # Reduce batch size
                    batch_size = max(1, batch_size // 2)
                    accumulation_steps = effective_batch_size // batch_size
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                            pin_memory=True)
                    print(f"Reduced batch size to {batch_size}, accumulation steps: {accumulation_steps}")
                    if batch_size == 1:
                        print(
                            "Batch size is already 1, cannot reduce further. Consider reducing model size or input dimensions.")
                else:
                    raise e

    # Save the model
    torch.save(model.state_dict(), 'multi_gpu_unet_model.pth')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root_dir', type=str, default='../data/brats18/train/combined/')
    parser.add_argument('--slice_range', type=tuple, default=(2, 150))
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    args = parser.parse_args()

    train(args)