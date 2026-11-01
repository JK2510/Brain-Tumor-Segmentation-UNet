import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from unet import UNet
from dataset import normalize_image

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = UNet(num_classes=checkpoint["num_classes"],
                 base_channels=checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = np.array(Image.open(args.image).convert("L"))
    normalized = normalize_image(image)
    tensor = torch.from_numpy(normalized).float()[None, None]
    tensor = torch.nn.functional.interpolate(
        tensor, size=(checkpoint["image_size"], checkpoint["image_size"]),
        mode="bilinear", align_corners=False
    ).to(device)
    with torch.no_grad():
        prediction = model(tensor).argmax(dim=1).squeeze().cpu().numpy()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(normalized, cmap="gray"); axes[0].set_title("MRI input")
    axes[1].imshow(prediction, cmap="viridis", vmin=0,
                   vmax=checkpoint["num_classes"] - 1); axes[1].set_title("Predicted classes")
    axes[2].imshow(normalized, cmap="gray")
    axes[2].imshow(prediction, cmap="viridis", alpha=0.45,
                   vmin=0, vmax=checkpoint["num_classes"] - 1)
    axes[2].set_title("Overlay")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Saved prediction visualization to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", default="models/best_model.pth")
    p.add_argument("--output", default="outputs/prediction.png")
    main(p.parse_args())
