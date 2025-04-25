"""
predict_multi_step.py
Generación de pronósticos multi-step para todo el turno.
Carga el modelo y scaler, itera recalcando lags y features de tiempo,
y predice la velocidad cada 30s durante todo el horizonte configurado.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import joblib
from tensorflow.keras.models import load_model

# --------------------------------------
# Configuración
# --------------------------------------
ROOT_DIR       = Path.cwd()
DATA_DIR       = ROOT_DIR / "data" / "processed" / "final"
MODELS_DIR     = ROOT_DIR / "models"
OUTPUT_DIR     = ROOT_DIR / "data" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros de pronóstico
horizon_hours  = 9  # Horas del turno
HORIZON_STEPS  = horizon_hours * 60 * 2  # pasos de 30s en int
LAGS           = [1, 2, 4, 10]          # en pasos de 30s
ROLL_WINDOWS   = [10, 20]               # en pasos de 30s

# Prefijos para archivos
FINAL_PATTERN  = "dataset_final_*.parquet"
SCALER_PATTERN = "scaler_*.pkl"
MODEL_PATTERN  = "model_*.h5"

# --------------------------------------
# Utilidades
def latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No se encontró archivos {pattern} en {directory}")
    return files[-1]

# Agregar features de tiempo a una fila DataFrame de un solo registro
def add_time_features(last_df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(last_df['_time'].iloc[0])
    last_df['hour']       = t.hour
    last_df['minute']     = t.minute
    last_df['dayofweek']  = t.dayofweek
    last_df['is_weekend'] = int(t.dayofweek >= 5)
    last_df['shift']      = int((t.hour >= 19) or (t.hour < 7))
    return last_df

# --------------------------------------
# Pipeline de predicción multi-step
# --------------------------------------
def predict_multi_step():
    # Carga scaler y modelo sin recompilar
    scaler_path = latest_file(MODELS_DIR, SCALER_PATTERN)
    model_path  = latest_file(MODELS_DIR, MODEL_PATTERN)
    scaler = joblib.load(scaler_path)
    model  = load_model(model_path, compile=False)
    print(f"[PREDICT] Usando scaler={scaler_path.name}, modelo={model_path.name}")

    # Cargar dataset final con features hasta el instante actual
    final_path = latest_file(DATA_DIR, FINAL_PATTERN)
    df_input = pd.read_parquet(final_path).sort_values('_time').reset_index(drop=True)
    print(f"[PREDICT] Datos iniciales cargados ({len(df_input)} registros)")

    results = []
    for step in range(int(HORIZON_STEPS)):
        # Obtiene la última fila
        last = df_input.iloc[-1:].copy()
        # Avanza timestamp 30s
        next_time = pd.to_datetime(last['_time'].iloc[0]) + pd.Timedelta(seconds=30)
        last['_time'] = next_time

        # Recalcular features de tiempo y lags
        last = add_time_features(last)
        for lag in LAGS:
            last[f'lag_{lag}'] = df_input['velocity_bpm'].iloc[-lag]
        for w in ROLL_WINDOWS:
            last[f'roll_mean_{w}'] = df_input['velocity_bpm'].iloc[-w:].mean()

        # Preparar X
        feature_cols = [c for c in last.columns if c not in ['_time', 'velocity_bpm', 'target', 'device_id']]
        X = last[feature_cols].values
        X_scaled = scaler.transform(X)

        # Predecir
        y_pred = model.predict(X_scaled, verbose=0)[0, 0]
        last['velocity_bpm'] = y_pred

        # Añadir a df_input y resultado
        df_input = pd.concat([df_input, last], ignore_index=True)
        results.append({'_time': next_time, 'predicted_velocity_bpm': y_pred})

    # Guardar todo el pronóstico
    out_file = OUTPUT_DIR / f"forecast_{datetime.date.today()}.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"[PREDICT] Pronóstico completo de {int(HORIZON_STEPS)} pasos guardado en: {out_file}")

if __name__ == '__main__':
    try:
        predict_multi_step()
    except Exception as e:
        print(f"[ERROR MULTI PREDICT] {e}")
        raise
