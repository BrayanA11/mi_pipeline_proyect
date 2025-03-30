from .model import load_model
from .preprocessing import preprocess_image
from .postprocessing import subclassify_prediction  # Importamos la función de subclasificación
from .config import CONFIG
import torch

class Detector:
    def __init__(self, device=None):
        if device is None:
            device = CONFIG["device"]
        self.device = device
        self.model = load_model(device=device)

    def predict(self, image_path):
        input_tensor = preprocess_image(image_path).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        # En lugar de la postprocesamiento original, usamos la subclasificación
        label, confidence = subclassify_prediction(output)
        return label, confidence
