import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import BrainMRIDataset
from unet import UNet
from metrics import dice_iou_per_class

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(Path(args.data_dir).rglob("*.h5")) + sorted(Path(args.data_dir).rglob("*.hdf5"))
    if not files:
        raise RuntimeError(f"No HDF5 files found in {args.data_dir}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = UNet(num_classes=checkpoint["num_classes"],
                 base_channels=checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(BrainMRIDataset(files, checkpoint["image_size"]),
                        batch_size=args.batch_size, shuffle=False)
    collected = {k: {"dice": [], "iou": []} for k in range(1, checkpoint["num_classes"])}
    with torch.no_grad():
        for images, masks in loader:
            metrics = dice_iou_per_class(model(images.to(device)), masks.to(device),
                                         checkpoint["num_classes"])
            for class_id, values in metrics.items():
                for metric_name, value in values.items():
                    collected[class_id][metric_name].append(value)
    for class_id, values in collected.items():
        print(f"Class {class_id}: Dice={np.mean(values['dice']):.4f}, "
              f"IoU={np.mean(values['iou']):.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--checkpoint", default="models/best_model.pth")
    p.add_argument("--batch_size", type=int, default=8)
    main(p.parse_args())
