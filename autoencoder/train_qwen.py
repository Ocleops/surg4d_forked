"""
Train Qwen feature autoencoder (inspired by splattalk)
"""

from pathlib import Path
import argparse
from typing import List, Literal
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from autoencoder.dataset import Autoencoder_dataset
from autoencoder.model_qwen import QwenAutoencoder


def train(
    data_dirs: List[Path],
    checkpoint_dir: Path,
    epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 256,
    num_workers: int = 8,
    device: str = 'cuda:0',
    full_dim: int = 3584,
    latent_dim: int = 3,
    sigma_input: float = 0.0,
    sigma_latent: float = 0.0,
    n_latent: int = 10,
    loss: Literal['mse', 'l1'] = 'mse',
) -> None:
    """Train Qwen feature autoencoder.

    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = Autoencoder_dataset(data_dirs)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    model = QwenAutoencoder(input_dim=full_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Create timestamped tensorboard log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_log_dir = checkpoint_dir / timestamp
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(tb_log_dir)

    global_step = 0
    best_val = float('inf')
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x = batch.to(device, dtype=torch.float32)
            x_std = x.std()
            if sigma_input > 0:
                x = x + torch.randn_like(x) * sigma_input

            z = model.encode(x)
            z_std = z.std()

            if sigma_latent > 0 and n_latent > 0:
                n_samples = z.shape[0]
                z = z.repeat(n_latent + 1, 1)
                z[n_samples:] += torch.randn_like(z[n_samples:]) * sigma_latent
                z = z / torch.linalg.norm(z, ord=2, dim=-1, keepdim=True).clamp(min=1e-8)
                x = x.repeat(n_latent + 1, 1)

            x_rec = model.decode(z)
            l1 = F.l1_loss(x_rec, x)
            mse = F.mse_loss(x_rec, x)
            train_loss = mse if loss == 'mse' else l1

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            tb.add_scalar(f'train/l1', l1.item(), global_step)
            tb.add_scalar(f'train/mse', mse.item(), global_step)
            tb.add_scalar('train/x_std', x_std.item(), global_step)
            tb.add_scalar('train/z_std', z_std.item(), global_step)
            global_step += 1
            pbar.set_postfix(loss=float(train_loss.item()))

        # Eval
        model.eval()
        l1_sum = 0.0
        mse_sum = 0.0
        n_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device, dtype=torch.float32)
                x_rec = model(x)
                l1_sum += F.l1_loss(x_rec, x)
                mse_sum += F.mse_loss(x_rec, x)
                n_samples += x.numel()
        tb.add_scalar(f'val/l1', l1_sum / n_samples, epoch)
        tb.add_scalar(f'val/mse', mse_sum / n_samples, epoch)
        val_loss = l1_sum / n_samples if loss == 'l1' else mse_sum / n_samples
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint_dir / 'best_ckpt.pth')

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"{epoch+1}_ckpt.pth")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--language_name', type=str, default='qwen_features')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--latent_dim', type=int, default=256)
    args = parser.parse_args()
    train(
        clip_path=args.dataset_path,
        lf_dir_names=[args.language_names],
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        latent_dim=args.latent_dim,
    )


if __name__ == '__main__':
    main()


