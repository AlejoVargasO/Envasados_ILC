"""
Generación de pronósticos
Carga el modelo y el scaler, toma la última fila de features,
predice la velocidad en el próximo paso (30s adelante) y guarda el resultado.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import joblib
from tensorflow.keras.models import load_model

# --------------------------------------
# Rutas y configuración
# --------------------------------------
ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data" / "processed" / "final"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "data" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prefijos para archivos
FINAL_PATTERN = "dataset_final_*.parquet"
SCALER_PATTERN = "scaler_*.pkl"
MODEL_PATTERN = "model_*.h5"

# --------------------------------------
# Funciones auxiliares
# --------------------------------------

def latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No se encontró archivos {pattern} en {directory}")
    return files[-1]

# --------------------------------------
# Pipeline de predicción
# --------------------------------------
def predict_next_step():
    # 1. Cargar scaler y modelo sin compilar (compile=False)
    scaler_path = latest_file(MODELS_DIR, SCALER_PATTERN)
    model_path = latest_file(MODELS_DIR, MODEL_PATTERN)
    print(f"[PREDICT] Cargando scaler {scaler_path.name} y modelo {model_path.name}")
    scaler = joblib.load(scaler_path)
    model = load_model(model_path, compile=False)

    # 2. Leer dataset con features
    final_path = latest_file(DATA_DIR, FINAL_PATTERN)
    print(f"[PREDICT] Cargando datos finales {final_path.name}")
    df = pd.read_parquet(final_path)

    # 3. Preparar X_last
    # Excluimos columnas no usadas: _time, velocity_bpm, target, device_id
    feature_cols = [c for c in df.columns if c not in ["_time", "velocity_bpm", "target", "device_id"]]
    last_row = df.iloc[-1:]
    X_last = last_row[feature_cols].values
    X_last_scaled = scaler.transform(X_last)

    # 4. Predecir siguiente paso
    y_pred = model.predict(X_last_scaled)
    next_velocity = float(y_pred.ravel()[0])
    next_time = pd.to_datetime(last_row['_time'].values[0]) + pd.Timedelta(seconds=30)

    # 5. Guardar pronóstico
    result = pd.DataFrame({
        "_time": [next_time],
        "predicted_velocity_bpm": [next_velocity]
    })
    out_file = OUTPUT_DIR / f"prediction_{datetime.date.today()}.csv"
    result.to_csv(out_file, index=False)
    print(f"[PREDICT] Pronóstico guardado en: {out_file}")

if __name__ == '__main__':
    try:
        predict_next_step()
    except Exception as e:
        print(f"[ERROR PREDICT] {e}")
        raise
