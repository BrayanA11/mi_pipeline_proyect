# preprocessing.py - Preprocesamiento de Datos
import torch
import torchvision.transforms as transforms
from PIL import Image
from config import CONFIG

def preprocess_image(image_path):
    """
    Carga y preprocesa una imagen para la detección de hojas.
    Aplica redimensionamiento, normalización y conversión a tensor.
    """
    transform = transforms.Compose([
        transforms.Resize(CONFIG["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(CONFIG["device"])
