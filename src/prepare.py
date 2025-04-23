"""
Prepare Feature Dataset
Lee el parquet fusionado (merge_quality_availability.py), genera columnas de
características (lags, rolling, time-features) y exporta un Parquet listo para
entrenamiento del modelo.
"""
from pathlib import Path
import pandas as pd
import datetime

# --------------------------
# Configuración de rutas
# --------------------------
PROC_DIR = Path("data/processed")
FINAL_DIR = PROC_DIR / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# Parámetros de feature engineering
# --------------------------
# Lags en pasos de 30s: 1=30s, 2=1min, 4=2min, 10=5min
LAGS = [1, 2, 4, 10]
# Rolling windows en número de pasos: 10=5min, 20=10min
ROLL_WINDOWS = [10, 20]

# --------------------------
# Utilidades
# --------------------------

def latest_parquet() -> Path:
    files = sorted(PROC_DIR.glob("merged_*.parquet"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("No hay archivos 'merged_*.parquet' en data/processed/")
    return files[-1]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["_time"].dt.hour
    df["minute"] = df["_time"].dt.minute
    df["dayofweek"] = df["_time"].dt.dayofweek  # 0=Lunes
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    # Turno: 0=mañana (7–19), 1=noche (19–7)
    df["shift"] = ((df["hour"] >= 19) | (df["hour"] < 7)).astype(int)
    return df


def add_lags(df: pd.DataFrame, col: str = "velocity_bpm") -> pd.DataFrame:
    for k in LAGS:
        df[f"lag_{k}"] = df[col].shift(k)
    return df


def add_rollings(df: pd.DataFrame, col: str = "velocity_bpm") -> pd.DataFrame:
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = df[col].rolling(window=w, min_periods=1).mean()
    return df

# --------------------------
# Pipeline principal
# --------------------------

def prepare():
    parquet_path = latest_parquet()
    print(f"[PREP] Cargando {parquet_path.name}")
    df = pd.read_parquet(parquet_path)

    # Asegurar orden temporal
    df = df.sort_values("_time").reset_index(drop=True)

    # 1. Features de tiempo
    df = add_time_features(df)

    # 2. Lags y rolling de velocidad
    df = add_lags(df)
    df = add_rollings(df)

    # 3. Borrar filas con NaN generados por lags (primeros N pasos)
    df = df.dropna().reset_index(drop=True)

    # 4. Guardar dataset final
    out_file = FINAL_DIR / f"dataset_final_{datetime.date.today()}.parquet"
    df.to_parquet(out_file, index=False)
    print(f"[PREP] Dataset final guardado en: {out_file}")


if __name__ == "__main__":
    try:
        prepare()
    except Exception as exc:
        print(f"[ERROR PREP] {exc}")
        raise
