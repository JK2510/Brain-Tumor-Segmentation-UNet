import numpy as np
import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
from src.unet import UNet
from src.dataset import normalize_image

st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")
st.title("Brain Tumor Segmentation using U-Net")
st.write("Upload a brain MRI image to generate a model prediction.")

@st.cache_resource
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    model = UNet(num_classes=checkpoint["num_classes"],
                 base_channels=checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, device

uploaded = st.file_uploader("Choose an MRI image", type=["png", "jpg", "jpeg", "tif", "tiff"])
checkpoint_path = st.text_input("Model checkpoint", "models/best_model.pth")

if uploaded:
    try:
        model, checkpoint, device = load_model(checkpoint_path)
        image = np.array(Image.open(uploaded).convert("L"))
        normalized = normalize_image(image)
        x = torch.from_numpy(normalized).float()[None, None]
        x = torch.nn.functional.interpolate(
            x, size=(checkpoint["image_size"], checkpoint["image_size"]),
            mode="bilinear", align_corners=False
        ).to(device)
        with torch.no_grad():
            mask = model(x).argmax(1).squeeze().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].imshow(normalized, cmap="gray"); axes[0].set_title("Input MRI")
        axes[1].imshow(mask, cmap="viridis", vmin=0,
                       vmax=checkpoint["num_classes"] - 1); axes[1].set_title("Predicted mask")
        axes[2].imshow(normalized, cmap="gray")
        axes[2].imshow(mask, cmap="viridis", alpha=0.45,
                       vmin=0, vmax=checkpoint["num_classes"] - 1)
        axes[2].set_title("Overlay")
        for ax in axes: ax.axis("off")
        st.pyplot(fig)
        st.caption("Research/educational demonstration only; not a medical diagnosis.")
    except FileNotFoundError:
        st.error("Model checkpoint not found. Train the model first.")
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
