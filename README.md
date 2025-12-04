# Proyecto de Segmentación y Predicción Temprana de Clientes con RFM, Machine Learning y Conformal Prediction

Este proyecto implementa un pipeline completo de analítica avanzada para segmentar clientes de un e-commerce y predecir desde su primera compra a qué tipo de cliente se convertirán, integrando metodologías de RFM, clustering, modelos supervisados y Conformal Prediction.

El objetivo es proporcionar a las áreas de marketing y CRM una herramienta accionable para priorizar inversiones en fidelización y mejorar la asignación de recursos.

---

## 1. Descripción General del Proyecto

El proyecto aborda los siguientes componentes:

- Segmentación de clientes utilizando el modelo RFM (Recency, Frequency, Monetary).
- Aplicación de métodos de clustering (K-Means y Fuzzy C-Means) con variables continuas y discretizadas.
- Entrenamiento de modelos supervisados para predecir el segmento futuro de un cliente después de su primera compra.
- Evaluación de Árboles de Decisión, Redes Neuronales y XGBoost.
- Implementación de Conformal Prediction para obtener conjuntos de predicción con garantías estadísticas.
- Construcción de un pipeline reproducible de análisis, entrenamiento y predicción.

El proyecto está basado en los hallazgos y metodología del documento técnico adjunto: “Modelos RFM y Conformal Prediction para la predicción temprana de clientes de alto valor”.

---

## 2. Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

```
Proyecto-Maestria-MIIA/
│
├── data/                    # Directorio para almacenamiento de datos.
│   ├── raw/                 # Datos originales exportados del e-commerce.
│   ├── processed/           # Datos limpios, transformados y finales.
│
├── notebooks/               # Notebooks Jupyter para análisis y pruebas.
│   ├── 01_EDA.ipynb         # Análisis exploratorio de datos.
│   ├── 02_RFM.ipynb         # Construcción de variables RFM.
│   ├── 03_Clustering.ipynb  # K-Means y Fuzzy C-Means.
│   ├── 04_Clasificacion.ipynb  # Entrenamiento de modelos supervisados.
│   ├── 05_Conformal.ipynb   # Evaluación con Conformal Prediction.
│
├── src/                     # Código fuente del proyecto.
│   ├── preprocessing.py     # Limpieza, codificación y transformación de datos.
│   ├── rfm.py               # Cálculo de Recency, Frequency y Monetary Value.
│   ├── clustering.py        # Implementación de K-Means y Fuzzy C-Means.
│   ├── classification.py    # Entrenamiento de modelos supervisados.
│   ├── conformal.py         # Capa de Conformal Prediction.
│   ├── utils.py             # Funciones auxiliares.
│
├── models/                  # Modelos entrenados y artefactos relacionados.
│   ├── xgboost_model.json   # Modelo final entrenado.
│   ├── scalers/             # Scalers y objetos de transformación.
│   ├── clusters/            # Resultados de clusterización.
│
├── results/                 # Resultados y reportes generados.
│   ├── figures/             # Visualizaciones (PCA, silueta, etc.).
│   ├── metrics/             # Reportes de evaluación.
│
├── requirements.txt         # Lista de dependencias del proyecto.
├── README.md                # Este archivo.
```

---

## 3. Cómo Ejecutar el Proyecto
### 3.1. Requisitos

Instalar dependencias:
```
pip install -r requirements.txt
```

Las principales librerías utilizadas incluyen:

- pandas
- numpy
- scikit-learn
- xgboost
- fcmeans
- matplotlib / seaborn
- joblib
- tqdm

---

### 3.2. Ejecución Paso a Paso
#### Paso 1. Preparación y Preprocesamiento de Datos

Desde `src/data/preprocessing.py`:
```python
from src.data.preprocessing import load_and_clean_data

df = load_and_clean_data("data/raw/datos.csv")
```
Este paso:
- Limpia datos
- Codifica variables categóricas
- Normaliza variables numéricas
- Genera dataset listo para modelado

---

#### Paso 2. Cálculo de Variables RFM
```python
from src.features.rfm import compute_rfm

rfm_scores = compute_rfm(df)
```
El módulo:
- Calcula Recency, Frequency y Monetary
- Los transforma a quintiles o reglas discretizadas
- Devuelve matriz continua y discretizada

---

#### Paso 3. Clusterización
**K-Means con variables discretizadas:**
```python
from src.clustering.kmeans_model import run_kmeans

kmeans_model, labels = run_kmeans(rfm_discretized, k=7)
```
**Fuzzy C-Means:**
```python
from src.clustering.fcm_model import run_fcm

fcm_model, memberships = run_fcm(rfm_continuous, c=6)
```

---

#### Paso 4. Entrenamiento de Modelos Supervisados

Desde `src/modeling/xgboost_model.py`:
```python
from src.modeling.xgboost_model import train_xgb

xgb_model = train_xgb(X_train, y_train)
```
También pueden ejecutarse:
- Árbol de Decisión (`decision_tree.py`)
- Red Neuronal MLP (`mlp_model.py`)

---

#### Paso 5. Aplicar Conformal Prediction
```python
from src.conformal.conformal_prediction import apply_conformal

prediction_sets = apply_conformal(model=xgb_model,
                                  X_test=X_test,
                                  y_test=y_test,
                                  alpha=0.1)
```
Este paso devuelve conjuntos de predicción por cliente con 90% de confianza.

---

#### Paso 6. Guardar Resultados
```python
joblib.dump(xgb_model, "outputs/models/xgboost.pkl")
prediction_sets.to_csv("outputs/predictions/prediction_sets.csv", index=False)
```

---

## 4. Resultados Principales
- El mejor modelo para clasificación de segmentos fue **XGBoost**, con:
  - Accuracy: 55.2 %
  - F1-Macro: 34.8 %
- Conformal Prediction generó conjuntos de predicción con tamaño promedio de 3.27 etiquetas.
- Sólo un 3 % de clientes presentaron predicciones deterministas.
- La mayoría de clientes con 4 etiquetas pertenecían a segmentos de alto potencial:
  - Nuevos / ticket alto
  - En riesgo (alto valor)
  - Leales recientes
  - Ocasionales antiguos

Esto permite priorizar presupuestos de marketing y mejorar retorno de inversión.

---

## 5. Cómo Interpretar la Salida Final

La salida final del sistema incluye:

- Segmento asignado por XGBoost
- Conjunto de posibles segmentos por Conformal Prediction
- Probabilidades por clase
- Nivel de incertidumbre del cliente

Ejemplo:

| Cliente | Predicción ML           | Prediction Set (90%) |
|---------|--------------------------|-----------------------|
| 12345   | Nuevo ticket alto        | {0,1,4,5}            |
| 67890   | Dormido bajo valor       | {1,3,5,6}            |

---

## 6. Trabajo Futuro
- Integrar modelos basados en secuencias (RNN/Transformers) cuando haya más historia.
- Incorporar datos de navegación web.
- Construir un dashboard interactivo para uso del área de marketing.
- Implementar explainability (SHAP) en predicciones individuales.
- Automatizar pipeline con Airflow o Prefect.

---

## 7. Referencia al Documento Técnico
La implementación corresponde íntegramente al documento técnico adjunto:  
**Modelos RFM y Conformal Prediction para la predicción temprana de clientes de alto valor**, utilizado como base conceptual y metodológica.

---

## 8. Contacto
Creado por José David Medina Lara  
Universidad de los Andes – Maestría en Inteligencia Analítica  
Correo: j.medinal@uniandes.edu.co