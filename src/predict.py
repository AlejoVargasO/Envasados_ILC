"""
src/predict.py
Script de pronóstico multi-step dinámico:
- Parámetros vía línea de comandos: --line y --hours
- Carga config para lags y roll_windows
- Lee último dataset final para la línea, genera device_idx
- Recarga scaler y feature_names para consistencia
- Ejecuta forecast por pasos de 30s recreando features
- Guarda CSV nombrado forecast_{line}_{hours}h_{YYYY-MM-DD}.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import joblib
from tensorflow.keras.models import load_model
from config import get_pipeline_config

# -----------------------------------
# Parse command-line arguments
# -----------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Pronóstico multi-step de velocidad de producción.")
    parser.add_argument('--line',  type=str, required=True, help='Línea de producción (ej. linea03)')
    parser.add_argument('--hours', type=int, required=True, help='Horizonte de predicción en horas')
    return parser.parse_args()

# -----------------------------------
# Utilidades
# -----------------------------------
def latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No se encontró archivo con patrón {pattern} en {directory}")
    return files[-1]

# -----------------------------------
# Agregar time features
# -----------------------------------
def add_time_features(df):
    t = pd.to_datetime(df['_time'].iloc[0])
    
    # Features básicos
    df['hour'] = t.hour
    df['minute'] = t.minute
    df['dayofweek'] = t.dayofweek
    df['is_weekend'] = int(t.dayofweek >= 5)
    
    # Features cíclicos
    df['hour_sin'] = np.sin(2 * np.pi * t.hour/24)
    df['hour_cos'] = np.cos(2 * np.pi * t.hour/24)
    df['minute_sin'] = np.sin(2 * np.pi * t.minute/60)
    df['minute_cos'] = np.cos(2 * np.pi * t.minute/60)
    df['dayofweek_sin'] = np.sin(2 * np.pi * t.dayofweek/7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * t.dayofweek/7)
    
    # Turno
    df['shift'] = 'day' if 7 <= t.hour < 19 else 'night'
    
    return df

# -----------------------------------
# Validar y limitar predicción
# -----------------------------------
def validate_prediction(pred, historical_data, max_change_percent=20):
    # Obtener estadísticas históricas
    mean_vel = historical_data['velocity_bpm'].mean()
    std_vel = historical_data['velocity_bpm'].std()
    min_vel = historical_data['velocity_bpm'].min()
    max_vel = historical_data['velocity_bpm'].max()
    
    # Límites absolutos basados en datos históricos
    lower_bound = max(0, min_vel * 0.5)  # No permitir menos del 50% del mínimo histórico
    upper_bound = max_vel * 1.2  # No permitir más del 120% del máximo histórico
    
    # Límites basados en la media y desviación estándar
    std_lower = mean_vel - 3 * std_vel
    std_upper = mean_vel + 3 * std_vel
    
    # Aplicar límites
    pred = np.clip(pred, lower_bound, upper_bound)
    pred = np.clip(pred, std_lower, std_upper)
    
    return float(pred)

# -----------------------------------
# Pronóstico multi-step
# -----------------------------------
def predict_multi_step(line: str, hours: int, lags: list, roll_windows: list):
    # Directorios
    ROOT_DIR   = Path.cwd()
    FINAL_DIR  = ROOT_DIR / 'data' / 'processed' / 'final'
    MODELS_DIR = ROOT_DIR / 'models'
    OUTPUT_DIR = ROOT_DIR / 'data' / 'predictions'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar scaler, modelo y feature_names
    scaler_path = latest_file(MODELS_DIR, 'scaler_*.pkl')
    feature_names_path = latest_file(MODELS_DIR, 'feature_names_*.pkl')
    model_path  = latest_file(MODELS_DIR, 'model_*.h5')

    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)
    model  = load_model(model_path, compile=False)
    print(f"[PREDICT] Usando scaler={scaler_path.name}, features={len(feature_names)}, modelo={model_path.name}")

    # Cargar dataset final filtrado
    parquet_pattern = f"dataset_final_*.parquet"
    df = pd.read_parquet(latest_file(FINAL_DIR, parquet_pattern))
    df = df[df['linea'] == line].sort_values('_time').reset_index(drop=True)
    
    # Mantener datos históricos para validación
    historical_data = df.copy()
    
    # Mantener o generar device_idx
    if 'device_idx' not in df.columns:
        df['device_idx'] = df['device_id'].astype('category').cat.codes

    # Forecast iterativo
    steps = hours * 60 * 2  # intervalos de 30s
    results = []
    last_pred = None

    for _ in range(steps):
        last = df.iloc[-1:].copy()
        next_time = pd.to_datetime(last['_time'].iloc[0]) + pd.Timedelta(seconds=30)
        last['_time'] = next_time

        # Time features, lags y rollings
        last = add_time_features(last)
        for lag in lags:
            last[f'lag_{lag}'] = df['velocity_bpm'].iloc[-lag]
        for w in roll_windows:
            last[f'roll_mean_{w}'] = df['velocity_bpm'].iloc[-w:].mean()

        # Seleccionar las mismas features que en entrenamiento
        X_df = last[feature_names]
        # Escalar
        X_scaled = scaler.transform(X_df)

        # Predicción
        y_pred = model.predict(X_scaled, verbose=0)[0,0]
        
        # Validar y limitar la predicción
        y_pred = validate_prediction(y_pred, historical_data)
        
        # Si es la primera predicción, guardarla
        if last_pred is None:
            last_pred = y_pred
        
        # Limitar el cambio porcentual entre predicciones consecutivas
        max_change = last_pred * 0.2  # máximo 20% de cambio
        y_pred = np.clip(y_pred, last_pred - max_change, last_pred + max_change)
        
        last['velocity_bpm'] = y_pred
        last_pred = y_pred

        # Agregar a df y resultados
        df = pd.concat([df, last], ignore_index=True)
        results.append({'_time': next_time, 'predicted_velocity_bpm': float(y_pred)})

    # Guardar CSV
    today = datetime.date.today().isoformat()
    out_file = OUTPUT_DIR / f"forecast_{line}_{hours}h_{today}.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"[PREDICT] Pronóstico de {steps} pasos guardado en: {out_file}")
    return out_file

# -----------------------------------
# Main
# -----------------------------------
if __name__ == '__main__':
    args = parse_args()
    cfg = get_pipeline_config()
    predict_multi_step(
        line=args.line,
        hours=args.hours,
        lags=cfg['lags'],
        roll_windows=cfg['roll_windows']
    )
