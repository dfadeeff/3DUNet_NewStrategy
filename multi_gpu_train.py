import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from model_UNet_2D_se_feat128 import UNet2D, CombinedLoss
from train_UNet_2D_se_feat128 import BrainMRI2DDataset, visualize_batch


def train(args):
    # Set up CUDA devices
    if torch.cuda.is_available():
        device_ids = list(range(torch.cuda.device_count()))
        print(f"Using {len(device_ids)} GPUs: {device_ids}")
    else:
        print("No GPUs available, using CPU")
        device_ids = None

    # Create dataset and dataloader
    dataset = BrainMRI2DDataset(args.train_root_dir, args.slice_range)
    dataloader = DataLoader(dataset, batch_size=args.batch_size * len(device_ids), shuffle=True, num_workers=4,
                            pin_memory=True)

    # Create the model
    model = UNet2D(in_channels=3, out_channels=1, init_features=32)

    if device_ids:
        model = DataParallel(model, device_ids=device_ids)
        model = model.cuda(device_ids[0])

    # Define loss function and optimizer
    criterion = CombinedLoss()
    if device_ids:
        criterion = criterion.cuda(device_ids[0])

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        model.train()
        for batch_idx, (inputs, targets, _) in enumerate(dataloader):
            if device_ids:
                inputs, targets = inputs.cuda(device_ids[0]), targets.cuda(device_ids[0])

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.num_epochs}], Step [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    # Save the model
    torch.save(model.state_dict(), 'multi_gpu_unet_model.pth')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root_dir', type=str, default='../data/brats18/train/combined/')
    parser.add_argument('--slice_range', type=tuple, default=(2, 150))
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    args = parser.parse_args()

    train(args)