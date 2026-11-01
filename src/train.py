import argparse, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from dataset import BrainMRIDataset
from unet import UNet
from losses import CombinedLoss
from metrics import dice_iou_per_class

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    files = sorted(Path(args.data_dir).rglob("*.h5")) + sorted(Path(args.data_dir).rglob("*.hdf5"))
    if len(files) < 2:
        raise RuntimeError(f"Need at least 2 .h5/.hdf5 files in {args.data_dir}; found {len(files)}")
    dataset = BrainMRIDataset(files, image_size=args.image_size)
    val_len = max(1, int(len(dataset) * args.val_split))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda"))

    model = UNet(num_classes=args.num_classes, base_channels=args.base_channels).to(device)
    criterion = CombinedLoss(args.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val = float("inf")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Device: {device} | Files: {len(files)} | Train: {train_len} | Validation: {val_len}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * images.size(0)

        model.eval()
        val_total = 0.0
        dice_totals = {k: [] for k in range(1, args.num_classes)}
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                val_total += criterion(logits, masks).item() * images.size(0)
                metrics = dice_iou_per_class(logits, masks, args.num_classes)
                for class_id, values in metrics.items():
                    dice_totals[class_id].append(values["dice"])

        train_loss = train_total / train_len
        val_loss = val_total / val_len
        dice_summary = {k: round(float(np.mean(v)), 4) if v else 0.0
                        for k, v in dice_totals.items()}
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} "
              f"| val_loss={val_loss:.4f} | val_dice={dice_summary}")

        if val_loss < best_val:
            best_val = val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "num_classes": args.num_classes,
                "base_channels": args.base_channels,
                "image_size": args.image_size,
                "best_val_loss": best_val,
            }
            torch.save(checkpoint, Path(args.output_dir) / "best_model.pth")
            print("  Saved new best model.")

    print(f"Training complete. Best validation loss: {best_val:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=240)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
