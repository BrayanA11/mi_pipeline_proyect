import torch
import pkg_resources

# Usando pkg_resources para obtener la ruta del archivo incluido en el paquete
MODEL_FILENAME = "swin_b_train_num0.pth"
MODEL_PATH = pkg_resources.resource_filename("mi_pipeline", MODEL_FILENAME)

CONFIG = {
    "model_path": MODEL_PATH,  # La ruta se obtiene dinámicamente
    "image_size": (224, 224),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "confidence_threshold": 0.5,
    "num_classes" :6,
}
