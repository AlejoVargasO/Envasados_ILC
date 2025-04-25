# Proyecto de Pronóstico de Producción – Planta de Envasados

[![CI](https://github.com/AlejoVargasO/Envasados_ILC/actions/workflows/ci.yml/badge.svg)](https://github.com/AlejoVargasO/Envasados_ILC/actions)

Este repositorio implementa un **sistema de pronóstico de velocidad de producción** (botellas/min) cada 30&nbsp;s para la planta de envasado de la Industria Licorera de Caldas. Incluye:

- **Pipeline de datos**: ingestión, limpieza & merge de CSV, generación de features.
- **Modelado**: entrenamiento de un MLP baseline (rápido de probar) con re-entrenamiento por turno.
- **API y programación**: Flask + APScheduler para orquestar ingest, merge, train & forecast.
- **Interfaz web**: página sencilla para que el operador solicite el pronóstico con un clic.
- **CI/CD**: GitHub Actions valida cada Pull Request y cada release (`v*.*.*`).

---

## 📁 Estructura de carpetas

```text
Proyecto_Envasados/
├── .github/                # Configuración de CI (GitHub Actions)
│   └── workflows/ci.yml
├── app.py                  # API Flask + scheduler de tareas
├── data/
│   ├── raw/                # CSV brutos descargados del servidor
│   ├── processed/          # Parquets intermedios y finales
│   │   └── final/
│   └── predictions/        # Resultados de predict.csv
├── models/                 # Modelos entrenados (.h5) y scalers (.pkl)
├── notebooks/              # Prototipos y EDA (Jupyter)
│   └── 01_exploracion.ipynb
├── requirements.txt        # Dependencias de Python
├── src/                    # Scripts modulares de pipeline
│   ├── ingest.py
│   ├── merge_quality_availability.py
│   ├── prepare.py
│   ├── train.py 
│   └── predict.py
├── templates/              # Plantillas HTML para Flask
│   └── index.html
└── README.md               # Este archivo
```

---

## 🚀 Comenzando (Instalación)

1. Clona el repositorio y entra en la carpeta:
   ```bash
   git clone https://github.com/TU_USUARIO/pronostico-produccion.git
   cd pronostico-produccion
   ```

2. Crea y activa el entorno virtual (Windows PowerShell):
   ```powershell
   python -m venv .venv
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```
   *(o en CMD: `activate.bat`)*

3. Instala dependencias:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. (Opcional) configura tu carpeta de red SMB o credenciales FTP en `src/ingest.py`.

---

## 🔄 Flujo de trabajo

### 1. Ingestión de datos
```bash
python src/ingest.py
```
Copiará los CSV de calidad y disponibilidad a `data/raw/` con sufijo de fecha.

### 2. Merge y limpieza
```bash
python src/merge_quality_availability.py
```
Genera `data/processed/merged_YYYY-MM-DD.parquet` con features básicas.

### 3. Feature engineering
```bash
python src/prepare.py
```
Añade lags, rolling means y time-features, guardando `final/dataset_final_YYYY-MM-DD.parquet`.

### 4. Entrenamiento del modelo
```bash
python src/train.py
```
Entrena un MLP baseline y guarda modelo y scaler en `models/`.

### 5. Predicción de un solo paso
```bash
python src/predict.py
```
Produce `data/predictions/prediction_YYYY-MM-DD.csv` con la velocidad a +30&nbsp;s.

---

## 🌐 API y Scheduler

Arranca la aplicación web y scheduler:
```bash
python app.py
```
- **Web**: `http://localhost:5000/` → interfaz para solicitar pronóstico.
- **Endpoints REST**:
  - `GET /ingest`  → lanza ingest
  - `GET /merge`   → lanza merge
  - `GET /train`   → lanza train
  - `GET /forecast`→ devuelve CSV de predict

El scheduler interno ejecuta ingest, merge, train y forecast automáticamente justo antes y después de cada turno (configurable en `app.py`).

---

## 🎯 Branching & Versiones (Git Flow)

Utiliza **Git Flow** para gestionar ramas:
1. **`feature/xxx`**: crea nuevas funcionalidades desde `develop`.
2. **`release/vX.Y.Z`**: prepara salida en `main`, etiqueta `vX.Y.Z`.
3. **`hotfix/vX.Y.Z`**: correcciones urgentes en `main`.

```bash
# Crear feature
git flow feature start predict-loop
# Finalizar feature
git flow feature finish predict-loop

# Crear release
git flow release start 1.0.0
# Finalizar release
git flow release finish 1.0.0
```

---

## 📦 Integración Continua (CI)

La carpeta `.github/workflows/ci.yml` contiene un workflow que se activa en:
- **Pull requests** a `develop` → lint, type-check y smoke tests.
- **Push de tags** `v*.*.*` → misma validación.

Ver badge de estado en la parte superior del README.

---

## 🤝 Contribuir

1. Forkea el repositorio.  
2. Crea una rama feature: `git flow feature start mi-feature`.  
3. Haz commits con mensajes claros (`feat`, `fix`, `chore`).  
4. Abre un Pull Request contra `develop`.  
5. Espera reviews, corrige según feedback y mergea.

---

## 📄 Licencia
📄 Licencia (Propietaria)

Copyright © 2025 Alejandro Vargas Orozco. Todos los derechos reservados.

Este código es propiedad de la Industria Licorera de Caldas y su uso, reproducción, distribución o modificación está estrictamente prohibido sin la autorización expresa del autor o la empresa.
