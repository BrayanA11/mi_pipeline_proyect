import torch
import torch.nn.functional as F

# Mapeo de índices a clases en español
IDX_TO_CLASS = {
    0: "Hoja saludable",
    1: "Complejo (combinación de clases, si corresponde)",
    2: "Óxido (Rust)",
    3: "Mancha ojo de rana (Frog Eye Leaf Spot)",
    4: "Mildiu polvoriento (Powdery Mildew)",
    5: "Sarna (Scab)",
}

# Umbral para considerar la hoja saludable (según la confianza)
THRESHOLD_HEALTHY = 0.5

def postprocess_output(output):
    """
    Aplica softmax a la salida del modelo, obtiene la clase predicha y la confianza.
    Retorna el índice de clase y la probabilidad máxima.
    """
    probabilities = F.softmax(output, dim=1)
    max_prob, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item(), max_prob.item()

def subclassify_prediction(output):
    """
    Dada la salida del modelo, aplica la lógica de subclasificación:
      - Si la confianza (max_prob) es menor al umbral o la clase predicha corresponde a "Hoja saludable",
        retorna "Hoja saludable".
      - En caso contrario, retorna "Hoja enferma".
    
    Retorna la etiqueta final y la confianza.
    """
    predicted_idx, max_prob = postprocess_output(output)
    # Si la confianza es baja o la clase predicha es "Hoja saludable", consideramos que es saludable.
    if max_prob < THRESHOLD_HEALTHY or IDX_TO_CLASS[predicted_idx] == "Hoja saludable":
        return "Hoja saludable", max_prob
    else:
        return "Hoja enferma", max_prob
