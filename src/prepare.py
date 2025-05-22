"""
src/prepare.py
Feature engineering dinámico:
- Carga el parquet fusionado más reciente para la línea configurada
- Filtra por línea
- Agrega lags y medias móviles según config.yaml
- Crea features de tiempo
- Convierte device_id a índice numérico
- Guarda dataset final listo para entrenamiento
"""
import pandas as pd
import glob
from pathlib import Path
import datetime
from config import get_pipeline_config

# -----------------------------------
# Configuración dinámica
# -----------------------------------
cfg = get_pipeline_config()
line         = cfg['line']
lags         = cfg['lags']
roll_windows = cfg['roll_windows']

# Directorios
PROC_DIR   = Path('data/processed')
FINAL_DIR  = PROC_DIR / 'final'
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# Obtener archivo más reciente
# -----------------------------------
def latest_parquet(pattern: str):
    files = sorted(
        glob.glob(pattern),
        key=lambda p: Path(p).stat().st_mtime
    )
    if not files:
        raise FileNotFoundError(f"No se encontró archivo parquet con patrón {pattern}")
    return Path(files[-1])

# -----------------------------------
# Función principal
# -----------------------------------
if __name__ == '__main__':
    # 1. Cargar último parquet merged para la línea
    pattern = str(PROC_DIR / f"merged_*.parquet")
    merged_path = latest_parquet(pattern)
    print(f"[PREP] Cargando {merged_path}")
    df = pd.read_parquet(merged_path)

    # 2. Filtrar por línea y reset index
    #df = df[df['linea'] == line].reset_index(drop=True)

    # 3. Ordenar por tiempo
    df = df.sort_values('_time').reset_index(drop=True)

    # 4. Convertir device_id a índice numérico (para embeddings)
    df['device_idx'] = df['device_id'].astype('category').cat.codes

    # 5. Añadir lags (pasos de 30s)
    for lag in lags:
        df[f'lag_{lag}'] = df['velocity_bpm'].shift(lag)

    # 6. Añadir rolling means
    for w in roll_windows:
        df[f'roll_mean_{w}'] = df['velocity_bpm'].rolling(window=w, min_periods=1).mean()

    # 7. Crear features de tiempo
    df['_time'] = pd.to_datetime(df['_time'])
    df['hour']       = df['_time'].dt.hour
    df['minute']     = df['_time'].dt.minute
    df['dayofweek']  = df['_time'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    # Turno: 'day' si entre 7 y 18, 'night' en otro caso
    df['shift'] = df['hour'].apply(lambda h: 'day' if 7 <= h < 19 else 'night')

    # 8. Eliminar filas con NaNs de lags
    min_lag = max(lags)
    df_final = df.iloc[min_lag:].reset_index(drop=True)

    # 9. Guardar dataset final
    today = datetime.date.today().isoformat()
    out_path = FINAL_DIR / f"dataset_final_{today}.parquet"
    df_final.to_parquet(out_path, index=False)
    print(f"[PREP] Dataset final guardado en: {out_path}")
