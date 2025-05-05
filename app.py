import streamlit as st
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.mobilenet import MobileNet_V3_Large_Weights
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set page config
st.set_page_config(page_title="Animal Detector using Deep Learning", layout="wide")
st.title("Animal Detection using Deep Learning")

# Class names
class_names = [
    "Bear", "Bull", "Cattle", "Cheetah", "Chicken", "Deer", "Duck", "Eagle", "Elephant",
    "Fox", "Frog", "Goat", "Goose", "Hamster", "Hedgehog", "Horse", "Jaguar", "Leopard", "Lion",
    "Lynx", "Magpie", "Monkey", "Mouse", "Mule", "Owl", "Parrot", "Pig", "Rabbit", "Raccoon",
    "Raven", "Rhinoceros", "Sheep", "Squirrel", "Swan", "Tiger", "Turkey"
]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
@st.cache_resource
def load_model():
    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        num_classes=37
    )
    model.load_state_dict(torch.load("fasterrcnn_models/fasterrcnn_best.pth", map_location=device))
    model.to(device).eval()
    return model

model = load_model()

# Two-column layout
left_col, right_col = st.columns([1, 2])

with left_col:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

with right_col:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        transform_img = F.to_tensor(image).to(device)

        with torch.no_grad():
            preds = model([transform_img])
        pred = preds[0]

        # Draw detections
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)

        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            if score < 0.3:
                continue
            x1, y1, x2, y2 = box.cpu()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor="red", facecolor="none")
            cls_name = class_names[label - 1]
            ax.text(x1, y1, f"{cls_name} {score:.2f}",
                    color="white", backgroundcolor="red", fontsize=8)
            ax.add_patch(rect)

        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("Upload an image to detect animals.")
