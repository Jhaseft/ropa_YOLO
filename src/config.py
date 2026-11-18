"""
Carga la configuración del proyecto (config.yaml) y las variables de entorno (.env).

Expone:
- RAIZ: ruta absoluta de la raíz del proyecto, para construir rutas desde
  cualquier script sin importar desde dónde se ejecute.
- CONFIG: diccionario con todo el contenido de config.yaml.
- ruta(): función para convertir una ruta relativa de config en absoluta.
"""

from pathlib import Path

import yaml
from dotenv import load_dotenv

# La raíz del proyecto es la carpeta que contiene a src/ (un nivel arriba de este archivo).
RAIZ = Path(__file__).resolve().parent.parent

# Carga las variables del archivo .env que está en la raíz del proyecto.
load_dotenv(RAIZ / ".env")


def cargar_config():
    """Lee config.yaml desde la raíz y lo devuelve como diccionario."""
    ruta_config = RAIZ / "config.yaml"
    if not ruta_config.exists():
        raise FileNotFoundError(
            f"No se encontró config.yaml en {ruta_config}. "
            "Debe estar en la raíz del proyecto."
        )
    with open(ruta_config, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Configuración disponible al importar el módulo.
CONFIG = cargar_config()


def ruta(relativa):
    """Convierte una ruta relativa (desde la raíz) en una ruta absoluta."""
    return RAIZ / relativa


# Rutas usadas con frecuencia, ya resueltas como absolutas.
RUTA_DATASET = ruta(CONFIG["rutas"]["dataset"])
RUTA_MODELO = ruta(CONFIG["rutas"]["modelo"])


if __name__ == "__main__":
    # Pequeña prueba manual: muestra la configuración cargada.
    print("Raíz del proyecto:", RAIZ)
    print("Dataset:", RUTA_DATASET)
    print("Modelo:", RUTA_MODELO)
    print("Modelo base:", CONFIG["modelo_base"])
    print("Entrenamiento:", CONFIG["entrenamiento"])
    print("Inferencia:", CONFIG["inferencia"])
