# Sistema de detección de defectos en prendas de vestir 👕

Proyecto universitario de **Inteligencia Artificial** (aprendizaje supervisado,
visión por computadora). Detecta defectos en prendas de vestir usando **YOLO**
de Ultralytics con *transfer learning*, y entrega un veredicto de control de
calidad: **APROBADA** o **RECHAZADA**.

Defectos que detecta (se ajustan según el dataset): **agujero, mancha,
costura_defectuosa, hilo_suelto**.

---

## Estructura del proyecto

```
deteccion-defectos-prendas/
├── config.yaml                 # Parámetros (rutas, entrenamiento, inferencia)
├── .env                        # Credenciales de Roboflow (NO se sube al repo)
├── .env.example                # Plantilla de variables de entorno
├── requirements.txt            # Dependencias de Python
├── data/
│   └── dataset/                # Dataset descargado de Roboflow (ignorado por git)
├── models/
│   └── best.pt                 # Mejor modelo entrenado (ignorado por git)
├── notebooks/
│   └── entrenamiento_colab.ipynb   # Entrenamiento en Colab con GPU
└── src/
    ├── config.py               # Carga config.yaml y .env
    ├── descargar_dataset.py    # Descarga el dataset de Roboflow
    ├── entrenar.py             # Entrena el modelo
    ├── evaluar.py              # Evalúa el modelo sobre test
    ├── detector.py             # Clase DetectorDefectos (inferencia)
    └── app.py                  # Aplicación web (Streamlit)
```

---

## Instalación en Windows

```powershell
# 1. Clonar el repositorio
git clone https://github.com/Jhaseft/ropa_YOLO.git
cd ropa_YOLO

# 2. Crear y activar el entorno virtual
python -m venv venv
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Configurar el dataset de Roboflow

El dataset viene de **Roboflow Universe**. Pasos recomendados:

1. Entra a un dataset de defectos de prendas en Roboflow Universe y haz **Fork**
   hacia tu workspace (ej. `jhasefts-workspace`).
2. En tu proyecto, ve a **Versions → Create New Version** para generar una versión.
3. Obtén tu **Private API Key** en Roboflow → **Settings → API Keys**
   (usa la clave privada, **no** la publishable `rf_...`).
4. El *slug* del proyecto sale de la URL:
   `app.roboflow.com/<WORKSPACE>/<PROJECT>/<VERSION>`.

Copia la plantilla y completa tus datos:

```powershell
copy .env.example .env
```

Edita `.env`:

```
ROBOFLOW_API_KEY=tu_api_key_privada
ROBOFLOW_WORKSPACE=jhasefts-workspace
ROBOFLOW_PROJECT=slug_del_proyecto
ROBOFLOW_VERSION=1
```

> También funciona con el workspace/proyecto del autor original de Universe
> (ej. `ROBOFLOW_WORKSPACE=sem7-project`, `ROBOFLOW_PROJECT=t-shirt_defect_dataset`)
> usando tu misma API key.

> 🔒 **Seguridad:** el archivo `.env` está en `.gitignore` y nunca debe subirse.

---

## Flujo de uso

### 1. Descargar el dataset

```powershell
python src/descargar_dataset.py
```

Descarga el dataset a `data/dataset/`, corrige las rutas de `data.yaml`, crea el
split `test` si no existe (~10% de train) y muestra un resumen.

### 2. Entrenar

**Opción A — Local (recomendado solo si tienes GPU):**

```powershell
python src/entrenar.py
# Sobrescribir parámetros:
python src/entrenar.py --epochs 100 --batch 8 --imgsz 512
```

**Opción B — Google Colab con GPU (recomendado):**

Abre `notebooks/entrenamiento_colab.ipynb` en Colab, activa la GPU
(`Entorno de ejecución → Cambiar tipo de entorno → GPU`) y ejecuta las celdas.
Al final descargas `best.pt` y lo copias a la carpeta `models/` del proyecto local.

El modelo entrenado se guarda en `runs/entrenamiento/exp/` y se copia a
`models/best.pt`.

### 3. Evaluar

```powershell
python src/evaluar.py
```

Evalúa sobre el split `test` y guarda las métricas (precisión, recall, F1,
mAP50, mAP50-95; global y por clase) en `runs/evaluacion/exp` y
`runs/evaluacion/metricas.json`. También genera la matriz de confusión y las curvas.

### 4. Ejecutar la aplicación

```powershell
streamlit run src/app.py
```

Abre la app en el navegador. Permite **subir una imagen** o usar la **cámara**,
muestra la imagen original y la anotada, la tabla de defectos, el conteo por tipo
y el veredicto (verde = APROBADA, rojo = RECHAZADA).

> Si no existe `models/best.pt`, la app avisa que primero hay que entrenar.

---

## Tecnologías

- **YOLO (Ultralytics)** — detección de objetos con transfer learning
- **Roboflow** — gestión y descarga del dataset
- **Streamlit** — interfaz web
- **PyTorch** — motor de entrenamiento
- **Python 3.11**
