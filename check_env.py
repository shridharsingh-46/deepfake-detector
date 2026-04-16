import sys
print("Python:", sys.version)

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except ImportError:
    print("PyTorch: NOT INSTALLED")

try:
    import numpy
    print("NumPy:", numpy.__version__)
except ImportError:
    print("NumPy: NOT INSTALLED")

try:
    import cv2
    print("OpenCV:", cv2.__version__)
except ImportError:
    print("OpenCV: NOT INSTALLED")

import os
datasets_path = "./datasets"
print("\nDatasets folder exists:", os.path.exists(datasets_path))
if os.path.exists(datasets_path):
    contents = os.listdir(datasets_path)
    print("Datasets contents:", contents if contents else "(empty)")

weights_path = "./training/weights"
print("Weights folder exists:", os.path.exists(weights_path))
if os.path.exists(weights_path):
    print("Weight files:", os.listdir(weights_path))
