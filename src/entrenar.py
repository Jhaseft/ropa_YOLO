"""
Entrena el modelo YOLO para detectar defectos en prendas (transfer learning).

- Parte del modelo base definido en config.yaml (yolo11n.pt).
- Usa los parámetros de config.yaml, pero se pueden sobrescribir por argumentos.
- Avisa si no hay GPU y recomienda el notebook de Colab.
- Guarda los resultados en runs/entrenamiento y copia best.pt a models/.

Uso:
    python src/entrenar.py
    python src/entrenar.py --epochs 100 --batch 8 --imgsz 512
"""

import argparse
import shutil

from config import CONFIG, RAIZ, RUTA_DATASET, RUTA_MODELO


def hay_gpu():
    """Devuelve True si hay una GPU disponible para PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def resolver_device(device):
    """
    Traduce el 'device' de config.yaml a un valor que YOLO acepte.
    Si es 'auto', usa la GPU 0 si existe, o 'cpu' en caso contrario.
    """
    if str(device).lower() == "auto":
        return 0 if hay_gpu() else "cpu"
    return device


def leer_argumentos():
    """Permite sobrescribir epochs/batch/imgsz desde la línea de comandos."""
    entrenamiento = CONFIG["entrenamiento"]
    parser = argparse.ArgumentParser(description="Entrenamiento YOLO de defectos en prendas")
    parser.add_argument("--epochs", type=int, default=entrenamiento["epochs"],
                        help="Número de épocas")
    parser.add_argument("--batch", type=int, default=entrenamiento["batch"],
                        help="Tamaño del batch")
    parser.add_argument("--imgsz", type=int, default=entrenamiento["imgsz"],
                        help="Tamaño de imagen")
    return parser.parse_args()


def main():
    args = leer_argumentos()
    entrenamiento = CONFIG["entrenamiento"]

    # Verifica que el dataset exista antes de empezar.
    if not RUTA_DATASET.exists():
        print(f"ERROR: no se encontró el dataset en {RUTA_DATASET}")
        print("Primero descarga el dataset con: python src/descargar_dataset.py")
        raise SystemExit(1)

    # Aviso sobre la GPU.
    if hay_gpu():
        print("GPU detectada: el entrenamiento será rápido.")
    else:
        print("ATENCIÓN: no se detectó GPU. En CPU el entrenamiento es MUY lento.")
        print("Se recomienda usar el notebook notebooks/entrenamiento_colab.ipynb "
              "(Google Colab con GPU gratis).")

    # Importamos YOLO aquí para no cargar la librería si solo se pedía ayuda.
    from ultralytics import YOLO

    print(f"\nCargando modelo base: {CONFIG['modelo_base']}")
    modelo = YOLO(CONFIG["modelo_base"])

    device = resolver_device(entrenamiento["device"])
    print("Iniciando entrenamiento con estos parámetros:")
    print(f"  epochs={args.epochs}  batch={args.batch}  imgsz={args.imgsz}  "
          f"patience={entrenamiento['patience']}  device={device}")

    # Los resultados se guardan en runs/entrenamiento/exp.
    modelo.train(
        data=str(RUTA_DATASET),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=entrenamiento["patience"],
        device=device,
        project=str(RAIZ / "runs" / "entrenamiento"),
        name="exp",
        exist_ok=True,
    )

    # Copia el mejor modelo a models/best.pt para usarlo luego.
    best = RAIZ / "runs" / "entrenamiento" / "exp" / "weights" / "best.pt"
    if best.exists():
        RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, RUTA_MODELO)
        print(f"\nEntrenamiento terminado. Modelo copiado a: {RUTA_MODELO}")
    else:
        print("\nEntrenamiento terminado, pero no se encontró best.pt. "
              "Revisa la carpeta runs/entrenamiento/exp/weights.")


if __name__ == "__main__":
    main()
