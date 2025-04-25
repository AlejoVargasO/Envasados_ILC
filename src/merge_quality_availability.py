"""
Merge Quality & Availability Pipeline
Lee los CSV de calidad y disponibilidad, limpia, mapea motivos de paro,
filtra sólo línea 3 (columna `linea`), pivota availability y crea un Parquet único listo para feature engineering.
"""
import pandas as pd
from pathlib import Path
import datetime

# ---------------------------------------
# Rutas
# ---------------------------------------
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Mapeo de stopping_reason a grupos operativos
CAUSE_MAP = {
    "FALLA MEC": "Mecanica",
    "FALLA ELEC": "Electrica",
    "ESPERA INS": "Insumo",
    "CAMBIO FORM": "CambioFormato",
    "-": "Otros",
    # Agrega más códigos según la tabla oficial
}

# Prefijos de archivo raw
CALIDAD_PREFIX = "calidad"
DISP_PREFIX = "disponibilidad"

# ---------------------------------------
# Funciones auxiliares
# ---------------------------------------

def latest_csv(prefix: str) -> Path:
    files = sorted(RAW_DIR.glob(f"{prefix}_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No se encontró archivo raw con prefijo '{prefix}' en {RAW_DIR}")
    return files[-1]

# ---------------------------------------
# Pipeline principal
# ---------------------------------------

def merge_and_clean():
    # 1. Cargar CSV
    calidad_path = latest_csv(CALIDAD_PREFIX)
    disponibilidad_path = latest_csv(DISP_PREFIX)
    print(f"[MERGE] Leyendo: {calidad_path.name}, {disponibilidad_path.name}")

    q = pd.read_csv(calidad_path)
    d = pd.read_csv(disponibilidad_path)

    # 2. Normalizar timestamp a pasos de 30 s
    q["_time"] = pd.to_datetime(q["_time"]).dt.floor("30s")
    d["_time"] = pd.to_datetime(d["_time"]).dt.floor("30s")

    # 3. Filtrar sólo datos de la línea 3 por columna 'linea'
    q = q[q['linea'] == 'linea03'].reset_index(drop=True)
    d = d[d['linea'] == 'linea03'].reset_index(drop=True)

    # 4. Eliminar columnas _field que no aportan
    q = q.drop(columns=[c for c in q.columns if c.startswith("_field")], errors="ignore")
    d = d.drop(columns=[c for c in d.columns if c.startswith("_field")], errors="ignore")

    # 5. Seleccionar columnas útiles
    q = q[["_time", "_value", "real_velocity", "product_id", "device_id"]]
    d = d[["_time", "device_id", "_value", "stopping_reason"]]

    # 6. Mapear estado y motivo de paro
    d["state_flag"] = (d["_value"] == "Produciendo").astype(int)
    d["stop_group"] = d["stopping_reason"].map(CAUSE_MAP).fillna("Otros")

    # 7. Limpiar duplicados en availability
    d = d.drop(columns=["_value", "stopping_reason"])
    d = d.drop_duplicates(subset=["_time", "device_id"], keep="first")

    # 8. Merge asof: unir calidad con availability más cercana por tiempo y dispositivo
    q = q.sort_values("_time")
    d = d.sort_values("_time")
    merged = pd.merge_asof(
        q, d,
        on="_time",
        by="device_id",
        direction="nearest",
        tolerance=pd.Timedelta("30s")
    )

    # 9. Rellenar NaNs de state_flag con 0
    merged["state_flag"] = merged["state_flag"].fillna(0).astype(int)

    # 10. Renombrar _value a velocity_bpm
    merged = merged.rename(columns={"_value": "velocity_bpm"})

    # 11. One-hot de stop_group
    merged = pd.get_dummies(merged, columns=["stop_group"], prefix="stop")

    # 12. Guardar Parquet
    out_path = PROC_DIR / f"merged_{datetime.date.today()}.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"[MERGE] Dataset fusionado (línea 3) guardado en: {out_path}")

if __name__ == "__main__":
    try:
        merge_and_clean()
    except Exception as exc:
        print(f"[ERROR MERGE] {exc}")
        raise
