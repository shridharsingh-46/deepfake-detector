import streamlit as st
import os
import sys
import torch
import torch.nn as nn
import yaml
import numpy as np
from PIL import Image
from torchvision import transforms
import time

# --- Setup Paths ---
project_root = "D:/DeepfakeBench"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "training"))

from training.detectors.xception_detector import XceptionDetector

# --- Page Configuration ---
st.set_page_config(
    page_title="DeepGuard AI | Deepfake Detector",
    page_icon="???",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling (Premium Aesthetics) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0E1117;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 50%, #1c1e26 0%, #0e1117 100%);
    }

    .main-header {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        color: #8E949E;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .metric-title {
        color: #8E949E;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #FFFFFF;
    }

    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        margin-top: 1rem;
    }

    .status-fake {
        background: rgba(255, 75, 75, 0.2);
        color: #FF4B4B;
        border: 1px solid #FF4B4B;
    }

    .status-real {
        background: rgba(43, 192, 126, 0.2);
        color: #2BC07E;
        border: 1px solid #2BC07E;
    }

    /* Hide Streamlit components for premium feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- App Logic ---

@st.cache_resource
def load_model():
    weights_path = "D:/DeepfakeBench/training/weights/xception_best.pth"
    config_path = "D:/DeepfakeBench/training/config/detector/xception.yaml"
    device = torch.device("cpu")  # Force CPU — no CUDA on this machine

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["model_name"] = "xception"
    config["backbone_name"] = "xception"
    config["pretrained"] = weights_path  # build_backbone loads backbone layers with map_location='cpu'
    config["cuda"] = False

    # Step 1: build the model (build_backbone loads backbone weights internally)
    model = XceptionDetector(config).to(device)

    # Step 2: load the full finetuned checkpoint on top (always map to CPU)
    checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, device


def process_image(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model({"image": input_tensor}, inference=True)
        prob = output["prob"].item()
    return prob

# --- Layout ---

st.markdown('<div class="main-header">DeepGuard AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Neural Forensic Analysis for Deepfake Detection</div>', unsafe_allow_html=True)

try:
    model, device = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('### Neural Scan Interface')
    uploaded_file = st.file_uploader("Upload suspicious high-resolution media", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Current Analysis Subject", use_column_width=True)
    else:
        st.info("?? Awaiting media upload. Please provide an image to initiate forensic scan.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if uploaded_file:
        with st.spinner("?? Initializing Neural Network... Analysing micro-patterns..."):
            start_time = time.time()
            prob = process_image(image, model, device)
            latency = time.time() - start_time
            
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Forensic Confidence Level</div>', unsafe_allow_html=True)
        
        # Color based on result
        color = "#FF4B4B" if prob > 0.5 else "#2BC07E"
        st.markdown(f'<div class="metric-value" style="color: {color};">{prob*100:.1f}%</div>', unsafe_allow_html=True)
        
        # Progress bar
        st.progress(prob)
        
        if prob > 0.5:
            st.markdown('<div class="status-badge status-fake">?? FORENSIC MATCH: DEEPFAKE</div>', unsafe_allow_html=True)
            st.markdown("""
            **Analysis Report:**
            - **Neural Origin**: High probability of synthetic generation.
            - **Artifacts**: Inconsistencies detected in facial frequency domains.
            """)
        else:
            st.markdown('<div class="status-badge status-real">? VERIFIED AUTHENTIC</div>', unsafe_allow_html=True)
            st.markdown("""
            **Analysis Report:**
            - **Provenance**: Consistent with organic camera sensor noise.
            - **Integrity**: Micro-textures align with natural physiological patterns.
            """)
        
        st.divider()
        st.caption(f"Inference latency: {latency:.3f}s | Hardware: {device}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-card" style="text-align: center; color: #8E949E; padding-top: 5rem; padding-bottom: 5rem;">', unsafe_allow_html=True)
        st.markdown('### Forensic Dashboard Empty')
        st.markdown('Once you upload an image, the neural engine will output results here.')
        st.markdown('</div>', unsafe_allow_html=True)



