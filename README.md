# clasficacion binaria de hojas
ejemplo de uso: codigo de prueba ....


//////////////

from mi_pipeline import Detector

#image_path = input("Ingresa la ruta a la imagen a analizar: ") # ruta de archivo en jpg
image_path = "/home/angel/imagenes/istockphoto-1342346355-612x612.jpg"    
detector = Detector(device="cpu")
label, conf = detector.predict(image_path)
print(f"Predicción: {label}")
print(f"Confianza: {conf*100:.2f}%")

////////////

