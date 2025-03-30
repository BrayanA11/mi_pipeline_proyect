import torch
from torchvision.models import swin_b
from config import CONFIG

def load_model(device=None):
    if device is None:
        device = CONFIG["device"]
    # Instanciar el modelo con el número correcto de clases
    model = swin_b(weights=None, num_classes=CONFIG["num_classes"])
    checkpoint = torch.load(CONFIG["model_path"], map_location=device)
    
    # Cargar los pesos. Si la estructura coincide (ahora el head es [6,1024]),
    # no debería haber error.
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model
