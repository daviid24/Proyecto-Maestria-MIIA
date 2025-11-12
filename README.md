# Proyecto: Clasificación de Comentarios Útiles

Este proyecto utiliza redes neuronales para predecir si un comentario de un producto es útil o no, basado en datos textuales y características del producto. Además, se comparan distintas configuraciones de la red para analizar su desempeño.

## Tabla de Contenidos

1. [Descripción](#descripción)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Requisitos Previos](#requisitos-previos)
4. [Instalación](#instalación)
5. [Uso](#uso)

---

## Descripción

El objetivo principal del proyecto es determinar si un comentario de Amazon es útil, utilizando técnicas de procesamiento de lenguaje natural (PLN) y redes neuronales. Los datos provienen de reseñas de productos de compras en Amazon, donde cada comentario tiene información como el texto, la categoría del producto y la puntuación asignada por el usuario.

Las redes neuronales implementadas permiten:
- Identificar comentarios útiles (clasificación binaria).
- Comparar distintas configuraciones de la red (Dropout, Batch Normalization, etc.).
- Almacenar y reutilizar modelos entrenados en formato `.h5`.

---

## Estructura del Proyecto

```plaintext
project_name/
├── data/
│   ├── raw/               # Datos originales
│   │   └── data_full.csv
│   ├── processed/         # Datos procesados
├── models/                # Modelos entrenados en formato .h5
├── scripts/
│   ├── main.py            # Script principal
│   ├── database_loader.py # Clase para cargar y procesar datos
│   ├── dataset_splitter.py# Clase para dividir los datos
│   ├── network.py         # Clase de la red neuronal
├── notebooks/             # Notebooks de experimentación (opcional)
├── requirements.txt       # Lista de dependencias
└── README.md              # Documentación del proyecto
```
---

## Requisitos Previos

Antes de ejecutar este proyecto, asegúrate de tener:
1. Python 3.8 o superior instalado.
2. Un entorno virtual configurado (usando conda o venv).
3. Las dependencias listadas en requirements.txt.

---

## Instalación

1. Clona el repositorio

```repositorio
git clone https://github.com/daviid24/amazon-useful-comments-classifier.git
cd amazon-useful-comments-classifier
```

2. Crea un entorno virtual e instala las dependencias:

```dependencias
conda create -n tf-m2 python=3.9
conda activate tf-m2
pip install -r requirements.txt
```

3. Asegúrate de que los datos estén en el directorio correcto:

```
Coloca el archivo data_full.csv en la carpeta data/raw/.
```

---

## Uso

1. Ejecutar script principal:

```ejecutar
python scripts/main.py
```

2. Salida esperada:
   - Modelos entrenados guardados en models/.
   - Gráficas de precisión y pérdida por configuración de la red.
   - Comparación de resultados entre modelos.
3. Modifica las configuraciones:
   - Puedes cambiar las configuraciones de la red (Dropout, Batch Normalization, etc.) desde main.py.

