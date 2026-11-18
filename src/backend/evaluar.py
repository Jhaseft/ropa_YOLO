"""
Evalúa el modelo entrenado (models/best.pt) sobre el split "test".

Calcula y guarda en runs/evaluacion/metricas.json:
- Métricas globales: precisión, recall, F1, mAP50 y mAP50-95.
- Las mismas métricas por cada clase.

También indica dónde quedaron la matriz de confusión y las curvas generadas.

Uso:
    python src/backend/evaluar.py
"""

import json

from config import CONFIG, RAIZ, RUTA_DATASET, RUTA_MODELO

CARPETA_EVAL = RAIZ / "runs" / "evaluacion"


def calcular_f1(precision, recall):
    """F1 = media armónica entre precisión y recall (0 si ambos son 0)."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    # Verifica que el modelo exista.
    if not RUTA_MODELO.exists():
        print(f"ERROR: no se encontró el modelo en {RUTA_MODELO}")
        print("Primero entrena con: python src/backend/entrenar.py")
        raise SystemExit(1)

    from ultralytics import YOLO

    print(f"Cargando modelo: {RUTA_MODELO}")
    modelo = YOLO(str(RUTA_MODELO))

    print("Evaluando sobre el split 'test'...")
    CARPETA_EVAL.mkdir(parents=True, exist_ok=True)
    resultados = modelo.val(
        data=str(RUTA_DATASET),
        split="test",
        iou=CONFIG["inferencia"]["iou"],
        conf=CONFIG["inferencia"]["confianza_minima"],
        project=str(CARPETA_EVAL),
        name="exp",
        exist_ok=True,
    )

    caja = resultados.box

    # Métricas globales (promedio de todas las clases).
    precision_global = float(caja.mp)
    recall_global = float(caja.mr)
    metricas = {
        "global": {
            "precision": precision_global,
            "recall": recall_global,
            "f1": calcular_f1(precision_global, recall_global),
            "mAP50": float(caja.map50),
            "mAP50-95": float(caja.map),
        },
        "por_clase": {},
    }

    # Métricas por clase. ap_class_index indica a qué clase corresponde cada fila.
    nombres = resultados.names
    for i, idx in enumerate(caja.ap_class_index):
        p = float(caja.p[i])
        r = float(caja.r[i])
        metricas["por_clase"][nombres[int(idx)]] = {
            "precision": p,
            "recall": r,
            "f1": calcular_f1(p, r),
            "mAP50": float(caja.ap50[i]),
            "mAP50-95": float(caja.ap[i]),
        }

    # Guarda el JSON.
    ruta_json = CARPETA_EVAL / "metricas.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    # Muestra las métricas en pantalla de forma legible.
    print("\n" + "=" * 50)
    print("MÉTRICAS GLOBALES")
    print("=" * 50)
    for nombre, valor in metricas["global"].items():
        print(f"  {nombre:10s}: {valor:.4f}")

    print("\nMÉTRICAS POR CLASE")
    for clase, valores in metricas["por_clase"].items():
        print(f"  {clase}:")
        for nombre, valor in valores.items():
            print(f"      {nombre:10s}: {valor:.4f}")

    print(f"\nMétricas guardadas en: {ruta_json}")
    print(f"Matriz de confusión y curvas (P, R, F1, PR) en: {resultados.save_dir}")


if __name__ == "__main__":
    main()
