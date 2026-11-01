# Detección, clasificación y conteo de granos de maíz con YOLOv11

Trabajo de titulación: *Evaluación de YOLOv11n y YOLOv11s para la detección, clasificación y conteo de granos de maíz sanos y con defectos*.

Los modelos detectan cada grano y lo clasifican en dos clases:
- **`Healthy`:** granos sanos (`Healthy` e `Immature`);
- **`Defect`:** granos con defectos (`Broken`, `Damage`, `Fungus`, `Shriveled` y `Weeveled`).

El conteo se obtiene de las detecciones.

## Estructura

```
maiz-yolo11-v2/
├── oe1/                 Objetivo específico 1: ingeniería de datos
│   ├── 1.x_*.ipynb      notebooks, en orden de ejecución
│   ├── comun.py         rutas del repositorio, lectura de datasets, partición, experimentos y análisis
│   ├── docs/            informe de cada actividad
│   ├── resultados/      partición por lote, registro de corridas, decisión
│   └── salida/          figuras de cada notebook
├── oe2/                 Objetivo específico 2: entrenamiento y ajuste de los modelos
│   ├── 2.x_*.ipynb      notebooks, en orden de ejecución
│   ├── comun_oe2.py     fondos, entrenamiento, registro de corridas, reglas de decisión y diagnóstico
│   ├── docs/            plan del OE2 e informe de cada actividad
│   ├── resultados/      experimentos/<fase>.csv, configuraciones adoptadas, entornos
│   └── salida/          figuras de cada notebook
├── nube/                notebooks para Google Colab (los zips de datos son locales)
├── fotos/               fotos propias (sin metadatos): hojas vacías para los fondos negativos y fotos de prueba con su referencia
├── explicabilidad/      pruebas de Grad-CAM con los modelos finales (exploratorio, sin informe)
├── antecedentes/        PIIC y figuras de la asignatura Proyectos Tecnológicos (julio de 2026)
├── datos/               (local) datasets originales y derivados
├── runs/                (local) corridas de Ultralytics
├── modelos/             (local) pesos preentrenados en COCO (yolo11n.pt, yolo11s.pt)
├── trabajo/             (local) archivos temporales de los pliegues
└── zenodo/             (local) paquetes para Zenodo
```

Git versiona el código, los notebooks, los informes, los resultados y las figuras. Las carpetas marcadas como *(local)* y los zips de `nube/` no se suben.

## Entorno

- **Instalación:** Python 3.14 y [uv](https://docs.astral.sh/uv/). `uv sync` crea `.venv/` con las versiones exactas de `uv.lock`: PyTorch 2.14.1 con CUDA 13.2, Ultralytics 8.4.174 y Albumentations 2.0.8.
- **Local:** preparación de datos, análisis y figuras, en una RTX 3060 Laptop de 6 GB.
- **Google Colab:** el entrenamiento, en una A100 de 40 GB.

## Datos

El dataset de partida es *Maize with prev dataset* (Roboflow Universe, licencia CC BY 4.0), re-anotado en varias versiones. Sus exportaciones se colocan en `datos/originales/`:

| Carpeta | Versión |
| :--- | :--- |
| `datos/originales/Data_original` | v0: exportación pública original |
| `datos/originales/Data` | v1: primera re-anotación |
| `datos/originales/Data_corr` | v2: curaduría de `Weeveled` |

Los notebooks generan el resto de `datos/`:
- `Data_corr_v6`: etiquetas corregidas (v3);
- `Data_2c_<resolución>_<variante>`: las 6 variantes del diseño factorial (640, 960 y 1280 px; `base` y `norm`);
- `etiquetas_v0`: la configuración elegida con las etiquetas originales (experimento complementario de limpieza);
- `Data_final`: dataset final (1280 px, normalización fotométrica, sin balanceo), partición por lote en `train/valid/test`;
- `fondos`: fondos negativos (impurezas y hojas vacías) con el mismo preprocesamiento, repartidos por lote de captura.

## Orden de ejecución

| Notebook | Qué hace | Informe |
| :--- | :--- | :--- |
| `oe1/1.1_eda.ipynb` | Análisis exploratorio del dataset original | `oe1/docs/1.1_eda.md` |
| `oe1/1.2_limpieza_etiquetas.ipynb` | Corrección de etiquetas (v0 → v3) | `oe1/docs/1.2_limpieza_etiquetas.md` |
| `oe1/1.2_ingenieria_datos.ipynb` | Consolidación a 2 clases, partición por lote, variantes del diseño factorial y paquete para la nube | `oe1/docs/1.2_ingenieria_datos.md` |
| `nube/1.2_experimentos_colab.ipynb` | Corridas del diseño factorial y de los experimentos complementarios en Colab | `oe1/docs/1.2_ingenieria_datos.md` |
| `oe1/1.2_experimentos.ipynb` | Integración de las corridas, análisis y decisión | `oe1/docs/1.2_ingenieria_datos.md` |
| `oe1/1.3_estructuracion_dataset.ipynb` | Construcción de `Data_final` y verificaciones | `oe1/docs/1.3_estructuracion_dataset.md` |
| `oe2/2.1_entorno_arquitecturas.ipynb` | Entorno, arquitecturas, transferencia y memoria | `oe2/docs/2.1_configuracion_entorno.md` |
| `oe2/2.1_hiperparametros.ipynb` | Análisis de los hiperparámetros con evidencias | `oe2/docs/2.1_configuracion_entorno.md` |
| `oe2/2.1_aumento_datos.ipynb` | Transformaciones de Albumentations, configuraciones de aumento y compatibilidad con la normalización | `oe2/docs/2.1_configuracion_entorno.md` |
| `oe2/2.2a_fondos_dataset.ipynb` | Dataset de fondos negativos y paquete del OE2 para la nube | `oe2/docs/2.2_entrenamiento.md` |
| `nube/2.2_entrenamiento_colab.ipynb` | Corridas de la 2.2 en Colab: fondos, cribado, confirmación y final | `oe2/docs/2.2_entrenamiento.md` |
| `oe2/2.2_entrenamiento.ipynb` | Integración de las corridas de la 2.2, análisis y decisión de cada fase | `oe2/docs/2.2_entrenamiento.md` |
| `oe2/2.2b_diagnostico_valid.ipynb` | Diagnóstico de los modelos finales en `valid`: tipo de error, subtipos, lotes, escenas, confianza e impurezas | `oe2/docs/2.2b_diagnostico_valid.md` |
| `oe2/2.2c_fotos_propias.ipynb` | Comportamiento con fotos propias de smartphone y exploración de factores | `oe2/docs/2.2c_fotos_propias.md` |

`oe2/docs/plan_oe2.md` resume el diseño del OE2 y las reglas de decisión, fijadas antes de entrenar.

## Entrenamiento en Google Colab

1. **En local**, la última sección de `oe1/1.2_ingenieria_datos.ipynb` (OE1) o de `oe2/2.2a_fondos_dataset.ipynb` (OE2) genera el paquete (`nube/oe1_nube.zip`, `nube/oe2_nube.zip`) con el código, la partición, los datos y los registros previos.
2. **El zip se sube a la raíz de Google Drive**, que solo se lee.
3. **En Colab** se abre el notebook de `nube/` con una A100, se eligen la fase, los modelos y las semillas de la sesión, y se ejecuta. Las corridas ya registradas se omiten.
4. **Al terminar**, la sesión descarga un zip con sus corridas y su registro. En el OE2, los zips de las fases anteriores se suben, sin cambiarles el nombre, a la raíz de Drive para que la fase siguiente lea sus decisiones. En local, los notebooks de análisis incorporan las corridas a `runs/` y sus filas al registro.

**Imágenes nuevas:** el dataset final está normalizado fotométricamente, así que toda imagen que no venga de él se prepara con `comun.preprocesar(img, 1280, 'norm')` antes de la inferencia.
