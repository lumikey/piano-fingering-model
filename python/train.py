"""
Train fingering prediction model using a Transformer.

One prediction per note (not per chord).

Input: 26 tokens x 5 features
  - Previous (5): midi_offset, time_since, black_key, token_type=0, unused=-1
  - Current (1): pitch_class, 0, black_key, token_type=0.5, unused=-1
  - Lookahead (20): midi_offset, time_until, black_key, token_type=1, finger_hint

finger_hint: -1 = unknown, 0-1 = normalized finger (0-4 -> 0-1)

Output: single finger prediction (0-4)
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import FingeringTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def mask_finger_hints(X_batch: torch.Tensor) -> torch.Tensor:
    """Randomly mask finger hints in lookahead tokens during training.

    For each sample, picks a random reveal_rate (0-1), then masks each
    lookahead finger hint with probability (1 - reveal_rate).

    Args:
        X_batch: (batch, 26, 5) input tokens

    Returns:
        X_batch with some finger hints (feature 4, tokens 6-25) masked to -1
    """
    batch_size = X_batch.shape[0]
    X_masked = X_batch.clone()

    # Random reveal rate per sample
    reveal_rates = torch.rand(batch_size, 1)  # (batch, 1)

    # Random mask for each lookahead token (indices 6-25 = 20 tokens)
    mask_probs = torch.rand(batch_size, 20)  # (batch, 20)

    # Mask where random > reveal_rate (i.e., don't reveal)
    should_mask = mask_probs > reveal_rates  # (batch, 20)

    # Apply mask to feature 4 of lookahead tokens (indices 6-25)
    X_masked[:, 6:26, 4] = torch.where(
        should_mask,
        torch.tensor(-1.0),
        X_masked[:, 6:26, 4]
    )

    return X_masked


def load_data(hand='right', val_split=0.2, seed=42):
    """Load fingering data from all sources."""
    data_dir = PROJECT_ROOT / "data"

    data = np.load(data_dir / f'fingering_data_{hand}.npz')
    X = data['X']  # (N, 16, 4)
    Y = data['Y']  # (N,)

    print(f"  Flowkey data: {len(X)} samples")

    # Add PIG dataset
    try:
        pig_data = np.load(data_dir / f'pig_fingering_data_{hand}.npz')
        X = np.concatenate([X, pig_data['X']], axis=0)
        Y = np.concatenate([Y, pig_data['Y']], axis=0)
        print(f"  + PIG data: {len(pig_data['X'])} samples")
    except FileNotFoundError:
        print("  PIG data not found")

    # Add scale data
    try:
        scale_data = np.load(data_dir / f'scale_fingering_data_{hand}.npz')
        X = np.concatenate([X, scale_data['X']], axis=0)
        Y = np.concatenate([Y, scale_data['Y']], axis=0)
        print(f"  + Scale data: {len(scale_data['X'])} samples")
    except FileNotFoundError:
        print("  Scale data not found")

    print(f"  Total: {len(X)} samples")

    # Shuffle
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    X, Y = X[indices], Y[indices]

    # Split
    n_val = int(len(X) * val_split)
    return {
        'X_train': torch.tensor(X[n_val:], dtype=torch.float32),
        'Y_train': torch.tensor(Y[n_val:], dtype=torch.long),
        'X_val': torch.tensor(X[:n_val], dtype=torch.float32),
        'Y_val': torch.tensor(Y[:n_val], dtype=torch.long),
    }


def train(hand='right', max_epochs=300, lr=0.001, d_model=64, nhead=4, num_layers=3,
          dim_feedforward=128, patience=50, lr_patience=15, lr_factor=0.5, min_lr=1e-6):
    """Train transformer fingering model."""
    print(f"Training {hand} hand fingering transformer...")

    data = load_data(hand)
    X_train, Y_train = data['X_train'], data['Y_train']
    X_val, Y_val = data['X_val'], data['Y_val']

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    model = FingeringTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers,
                                  dim_feedforward=dim_feedforward)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: d_model={d_model}, heads={nhead}, layers={num_layers}, ff={dim_feedforward} ({n_params:,} params)")

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    lr_epochs_without_improvement = 0
    current_lr = lr

    for epoch in range(max_epochs):
        # Train
        model.train()

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()

            # Randomly mask finger hints for data augmentation
            X_batch = mask_finger_hints(X_batch)

            logits = model(X_batch)  # (batch, 5)
            loss = F.cross_entropy(logits, Y_batch, label_smoothing=0.1)

            loss.backward()
            optimizer.step()

        # Validate every 5 epochs (with no hints revealed - common case)
        if epoch % 5 == 0 or epoch == max_epochs - 1:
            model.eval()
            with torch.no_grad():
                # Mask all finger hints for validation
                X_val_masked = X_val.clone()
                X_val_masked[:, 6:26, 4] = -1.0

                logits = model(X_val_masked)
                preds = logits.argmax(dim=1)
                val_acc = (preds == Y_val).float().mean().item()
        else:
            val_acc = best_val_acc

        # Check for improvement
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            lr_epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            lr_epochs_without_improvement += 1

        # LR scheduling
        if lr_epochs_without_improvement >= lr_patience and current_lr > min_lr:
            current_lr = max(current_lr * lr_factor, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            lr_epochs_without_improvement = 0
            print(f"  Epoch {epoch+1}: Reduced LR to {current_lr:.2e}")

        # Progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: val_acc={val_acc:.4f}, best={best_val_acc:.4f}")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    print(f"\n  Best val_acc: {best_val_acc:.4f}")

    # Per-finger accuracy (with no hints revealed)
    model.eval()
    with torch.no_grad():
        X_val_masked = X_val.clone()
        X_val_masked[:, 6:26, 4] = -1.0

        logits = model(X_val_masked)
        preds = logits.argmax(dim=1)

        print("  Per-finger accuracy:")
        for finger in range(5):
            mask = Y_val == finger
            if mask.sum() > 0:
                acc = (preds[mask] == Y_val[mask]).float().mean().item()
                print(f"    Finger {finger+1}: {acc:.4f} ({mask.sum().item()} samples)")

    # Save
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    model_path = checkpoints_dir / f"fingering_transformer_{hand}.pt"
    torch.save({
        'model_state': model.state_dict(),
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'dim_feedforward': dim_feedforward,
        'val_acc': best_val_acc
    }, model_path)
    print(f"\n  Saved: {model_path}")

    return model, best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train fingering transformer')
    parser.add_argument('--hand', choices=['left', 'right', 'both'], default='both',
                        help='Which hand to train (default: both)')
    parser.add_argument('--d_model', type=int, default=64,
                        help='Transformer embedding dimension (default: 64)')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer layers (default: 3)')
    parser.add_argument('--dim_feedforward', type=int, default=128,
                        help='Feedforward dimension (default: 128)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Max epochs (default: 300)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')

    args = parser.parse_args()

    hands = ['left', 'right'] if args.hand == 'both' else [args.hand]

    for hand in hands:
        train(hand,
              max_epochs=args.epochs,
              lr=args.lr,
              d_model=args.d_model,
              nhead=args.nhead,
              num_layers=args.num_layers,
              dim_feedforward=args.dim_feedforward,
              patience=args.patience)
        print()
