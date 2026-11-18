"""
Aplicación web (Streamlit) para detectar defectos en prendas de vestir.

Permite subir una imagen o tomar una foto con la cámara, muestra la imagen
original y la anotada, una tabla de defectos, el conteo por tipo y un veredicto
de control de calidad (APROBADA / RECHAZADA).

Uso:
    streamlit run src/frontend/app.py
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# El frontend vive en src/frontend/, pero la lógica (detector, config) está en
# src/backend/. Agregamos esa carpeta al path para poder importarla.
RUTA_BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(RUTA_BACKEND))

from config import CONFIG, RUTA_MODELO
from detector import DetectorDefectos

# Configuración general de la página.
st.set_page_config(page_title="Detección de defectos en prendas", page_icon="👕", layout="wide")


@st.cache_resource
def cargar_detector(confianza):
    """Carga el detector una sola vez (se re-crea solo si cambia la confianza)."""
    return DetectorDefectos(confianza=confianza)


def mostrar_resultados(detector, imagen):
    """Ejecuta la detección y pinta los resultados en la interfaz."""
    imagen_anotada, defectos, veredicto = detector.detectar(imagen)

    # Imagen original y anotada lado a lado.
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagen original")
        st.image(imagen, use_container_width=True)
    with col2:
        st.subheader("Defectos detectados")
        st.image(imagen_anotada, use_container_width=True)

    # Veredicto con color (verde = aprobada, rojo = rechazada).
    if veredicto == "APROBADA":
        st.success(f"✅ Veredicto: {veredicto} (no se detectaron defectos)")
    else:
        st.error(f"❌ Veredicto: {veredicto} ({len(defectos)} defecto(s) detectado(s))")

    if defectos:
        # Tabla de defectos detectados.
        st.subheader("Detalle de defectos")
        tabla = pd.DataFrame([
            {
                "Clase": d["clase"],
                "Confianza": f"{d['confianza']:.2%}",
                "Caja (x1, y1, x2, y2)": d["caja"],
            }
            for d in defectos
        ])
        st.dataframe(tabla, use_container_width=True)

        # Conteo por tipo de defecto.
        st.subheader("Conteo por tipo")
        conteo = Counter(d["clase"] for d in defectos)
        st.bar_chart(pd.Series(conteo, name="cantidad"))


def main():
    st.title("👕 Sistema de detección de defectos en prendas de vestir")
    st.write(
        "Proyecto de Inteligencia Artificial (visión por computadora) que usa "
        "YOLO para encontrar defectos como agujeros, manchas, costuras "
        "defectuosas o hilos sueltos, y da un veredicto de control de calidad."
    )

    # Si no hay modelo entrenado, avisamos y no seguimos.
    if not RUTA_MODELO.exists():
        st.warning(
            "⚠️ No se encontró el modelo entrenado (models/best.pt).\n\n"
            "Primero hay que entrenar el modelo:\n"
            "1. `python src/backend/descargar_dataset.py`\n"
            "2. `python src/backend/entrenar.py` (o usar el notebook de Colab)"
        )
        st.stop()

    # Sidebar con el control de confianza mínima.
    st.sidebar.header("Configuración")
    confianza = st.sidebar.slider(
        "Confianza mínima",
        min_value=0.05,
        max_value=0.95,
        value=float(CONFIG["inferencia"]["confianza_minima"]),
        step=0.05,
        help="Solo se muestran defectos con una confianza mayor o igual a este valor.",
    )

    detector = cargar_detector(confianza)

    # Dos pestañas: subir imagen o usar la cámara.
    pestana_subir, pestana_camara = st.tabs(["📁 Subir imagen", "📷 Cámara"])

    with pestana_subir:
        archivo = st.file_uploader(
            "Sube una foto de la prenda", type=["jpg", "jpeg", "png", "webp"]
        )
        if archivo is not None:
            imagen = Image.open(archivo).convert("RGB")
            mostrar_resultados(detector, imagen)

    with pestana_camara:
        foto = st.camera_input("Toma una foto de la prenda")
        if foto is not None:
            imagen = Image.open(foto).convert("RGB")
            mostrar_resultados(detector, imagen)


if __name__ == "__main__":
    main()
