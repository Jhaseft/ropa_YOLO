"""
Descarga el dataset de defectos en prendas desde Roboflow y prepara data.yaml.

Pasos que realiza:
1. Lee las variables ROBOFLOW_* del .env y valida que existan (sin mostrar
   valores sensibles como la API key).
2. Descarga el dataset en formato "yolov8" dentro de data/dataset/.
3. Reescribe data.yaml con rutas absolutas y correctas (train/valid/test).
4. Si el dataset no trae split "test", lo crea moviendo ~10% de train.
5. Imprime un resumen: clases y cantidad de imágenes por split.

Uso:
    python src/descargar_dataset.py
"""

import os
import shutil
from pathlib import Path

import yaml

from config import RAIZ, RUTA_DATASET

# Carpeta destino donde quedará el dataset (data/dataset/).
CARPETA_DATASET = RUTA_DATASET.parent


def validar_variables():
    """Comprueba que las variables de Roboflow existan. Devuelve un dict con ellas."""
    requeridas = {
        "ROBOFLOW_API_KEY": "la Private API Key (Roboflow -> Settings -> API Keys)",
        "ROBOFLOW_WORKSPACE": "el workspace (ej. jhasefts-workspace)",
        "ROBOFLOW_PROJECT": "el slug del proyecto (sale de la URL del proyecto)",
        "ROBOFLOW_VERSION": "el número de versión del dataset (ej. 1)",
    }

    valores = {}
    faltantes = []
    for nombre, ayuda in requeridas.items():
        valor = os.getenv(nombre)
        if not valor:
            faltantes.append(f"  - {nombre}: falta {ayuda}")
        else:
            valores[nombre] = valor

    if faltantes:
        print("ERROR: faltan variables en el archivo .env:\n")
        print("\n".join(faltantes))
        print(
            "\nRevisa el archivo .env en la raíz del proyecto "
            "(puedes guiarte con .env.example)."
        )
        raise SystemExit(1)

    # Confirmación sin exponer la API key.
    print("Variables de Roboflow detectadas correctamente:")
    print(f"  Workspace: {valores['ROBOFLOW_WORKSPACE']}")
    print(f"  Proyecto:  {valores['ROBOFLOW_PROJECT']}")
    print(f"  Versión:   {valores['ROBOFLOW_VERSION']}")
    print("  API key:   (oculta por seguridad)")
    return valores


def descargar_desde_roboflow(valores):
    """Descarga el dataset usando la librería de Roboflow. Devuelve la carpeta."""
    # Se importa aquí para que el resto del script no dependa de roboflow
    # hasta que realmente se necesite.
    from roboflow import Roboflow

    print("\nConectando con Roboflow y descargando el dataset...")
    rf = Roboflow(api_key=valores["ROBOFLOW_API_KEY"])
    proyecto = rf.workspace(valores["ROBOFLOW_WORKSPACE"]).project(
        valores["ROBOFLOW_PROJECT"]
    )
    version = proyecto.version(int(valores["ROBOFLOW_VERSION"]))

    # IMPORTANTE: NO se crea la carpeta destino antes de descargar. Si la carpeta
    # ya existe, Roboflow asume que el dataset ya está bajado y NO extrae nada.
    # Por eso borramos cualquier descarga previa y usamos overwrite=True.
    if CARPETA_DATASET.exists():
        shutil.rmtree(CARPETA_DATASET)

    # Descarga en formato yolov8 directamente en data/dataset/.
    dataset = version.download("yolov8", location=str(CARPETA_DATASET), overwrite=True)
    print(f"Dataset descargado en: {dataset.location}")

    # Localiza el data.yaml real: a veces Roboflow lo deja en una subcarpeta.
    carpeta = Path(dataset.location)
    if not (carpeta / "data.yaml").exists():
        encontrados = list(carpeta.rglob("data.yaml"))
        if not encontrados:
            print("ERROR: la descarga terminó pero no se encontró data.yaml.")
            print("Revisa el workspace/proyecto/versión en el .env e inténtalo de nuevo.")
            raise SystemExit(1)
        carpeta = encontrados[0].parent

    return carpeta


def _contar_imagenes(carpeta_split):
    """Cuenta las imágenes dentro de <split>/images."""
    carpeta_img = carpeta_split / "images"
    if not carpeta_img.exists():
        return 0
    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in carpeta_img.iterdir() if p.suffix.lower() in extensiones)


def crear_split_test(carpeta):
    """Si no hay split test, lo crea moviendo ~10% de las imágenes de train."""
    carpeta_test = carpeta / "test"
    carpeta_train = carpeta / "train"

    if _contar_imagenes(carpeta_test) > 0:
        return  # ya existe test con imágenes, no hacemos nada

    print("\nEl dataset no trae split 'test'. Creando uno con ~10% de train...")
    (carpeta_test / "images").mkdir(parents=True, exist_ok=True)
    (carpeta_test / "labels").mkdir(parents=True, exist_ok=True)

    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imagenes = sorted(
        p for p in (carpeta_train / "images").iterdir()
        if p.suffix.lower() in extensiones
    )
    # Tomamos 1 de cada 10 imágenes para el test (al menos 1 si hay imágenes).
    cantidad = max(1, len(imagenes) // 10) if imagenes else 0

    for img in imagenes[:cantidad]:
        etiqueta = carpeta_train / "labels" / (img.stem + ".txt")
        # Mueve la imagen.
        shutil.move(str(img), str(carpeta_test / "images" / img.name))
        # Mueve su etiqueta si existe.
        if etiqueta.exists():
            shutil.move(str(etiqueta), str(carpeta_test / "labels" / etiqueta.name))

    print(f"Se movieron {cantidad} imágenes de train a test.")


def reescribir_data_yaml(carpeta):
    """Reescribe data.yaml con rutas absolutas correctas, conservando las clases."""
    ruta_yaml = carpeta / "data.yaml"
    with open(ruta_yaml, "r", encoding="utf-8") as f:
        datos = yaml.safe_load(f)

    # Rutas absolutas basadas en la carpeta real del dataset.
    datos["path"] = str(carpeta.resolve())
    datos["train"] = "train/images"
    datos["val"] = "valid/images"
    datos["test"] = "test/images"

    with open(ruta_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(datos, f, allow_unicode=True, sort_keys=False)

    return datos


def imprimir_resumen(carpeta, datos):
    """Muestra las clases y la cantidad de imágenes por split."""
    print("\n" + "=" * 50)
    print("RESUMEN DEL DATASET")
    print("=" * 50)

    nombres = datos.get("names", [])
    print(f"Clases ({len(nombres)}): {nombres}")

    for split in ("train", "valid", "test"):
        total = _contar_imagenes(carpeta / split)
        print(f"  {split:6s}: {total} imágenes")

    print(f"\ndata.yaml listo en: {carpeta / 'data.yaml'}")
    print("Ya puedes entrenar con: python src/entrenar.py")


def main():
    valores = validar_variables()
    carpeta = descargar_desde_roboflow(valores)
    crear_split_test(carpeta)
    datos = reescribir_data_yaml(carpeta)
    imprimir_resumen(carpeta, datos)


if __name__ == "__main__":
    main()
