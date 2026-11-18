"""
Clase DetectorDefectos: envuelve el modelo YOLO para detectar defectos en una
imagen y dar un veredicto de control de calidad.

Acepta imágenes como ruta, objeto PIL o arreglo numpy, y devuelve:
- La imagen anotada en RGB (con las cajas dibujadas).
- La lista de defectos: [{clase, confianza, caja}].
- El veredicto: "APROBADA" si no hay defectos, "RECHAZADA" si hay.
"""

import numpy as np
from PIL import Image

from config import CONFIG, RUTA_MODELO


class DetectorDefectos:
    def __init__(self, pesos=None, confianza=None, iou=None):
        """
        pesos: ruta a los pesos .pt (por defecto models/best.pt de config).
        confianza: umbral de confianza mínima (por defecto el de config).
        iou: umbral de IoU para NMS (por defecto el de config).
        """
        from ultralytics import YOLO

        self.pesos = str(pesos) if pesos else str(RUTA_MODELO)
        self.confianza = confianza if confianza is not None else CONFIG["inferencia"]["confianza_minima"]
        self.iou = iou if iou is not None else CONFIG["inferencia"]["iou"]
        self.modelo = YOLO(self.pesos)

    def _preparar_imagen(self, imagen):
        """Convierte la entrada (ruta, PIL o numpy) a un arreglo numpy RGB."""
        if isinstance(imagen, (str,)):
            return np.array(Image.open(imagen).convert("RGB"))
        if isinstance(imagen, Image.Image):
            return np.array(imagen.convert("RGB"))
        if isinstance(imagen, np.ndarray):
            return imagen
        raise TypeError("La imagen debe ser una ruta, un objeto PIL o un arreglo numpy.")

    def detectar(self, imagen):
        """
        Ejecuta la detección sobre la imagen.

        Devuelve una tupla:
            (imagen_anotada_rgb, lista_defectos, veredicto)
        """
        imagen_rgb = self._preparar_imagen(imagen)

        # IMPORTANTE: cuando Ultralytics recibe un arreglo numpy, asume que
        # viene en orden BGR (estilo OpenCV). Nuestra imagen está en RGB, así que
        # la convertimos a BGR para que el modelo vea los colores correctos.
        imagen_bgr = imagen_rgb[:, :, ::-1]
        resultados = self.modelo.predict(
            source=imagen_bgr,
            conf=self.confianza,
            iou=self.iou,
            verbose=False,
        )
        resultado = resultados[0]

        # Construye la lista de defectos detectados.
        defectos = []
        nombres = resultado.names
        for caja in resultado.boxes:
            clase_id = int(caja.cls[0])
            defectos.append({
                "clase": nombres[clase_id],
                "confianza": float(caja.conf[0]),
                # Caja en formato [x1, y1, x2, y2] (coordenadas de píxeles).
                "caja": [round(float(v), 1) for v in caja.xyxy[0].tolist()],
            })

        # plot() devuelve la imagen anotada en BGR (estilo OpenCV); la pasamos a RGB.
        anotada_bgr = resultado.plot()
        imagen_anotada = anotada_bgr[:, :, ::-1]

        # Veredicto de control de calidad.
        veredicto = "RECHAZADA" if defectos else "APROBADA"

        return imagen_anotada, defectos, veredicto


if __name__ == "__main__":
    # Prueba rápida desde consola: python src/backend/detector.py ruta/a/imagen.jpg
    import sys

    if len(sys.argv) < 2:
        print("Uso: python src/backend/detector.py <ruta_imagen>")
        raise SystemExit(1)

    detector = DetectorDefectos()
    _, defectos, veredicto = detector.detectar(sys.argv[1])
    print(f"Veredicto: {veredicto}")
    print(f"Defectos encontrados: {len(defectos)}")
    for d in defectos:
        print(f"  - {d['clase']} (confianza {d['confianza']:.2f}) caja {d['caja']}")
