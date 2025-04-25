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

# Mapeo completo de stopping_reason a grupos operativos
CAUSE_MAP = {
    # Mantenimiento
    "M01": "Mantenimiento_Averia",
    "M02": "Mantenimiento_Ajuste",
    "M03": "Mantenimiento_Servicios",
    "M04": "Mantenimiento_Programado",
    "M05": "Mantenimiento_Planeado",
    "M06": "Mantenimiento_AjusteFinalCambioFormato",
    "M09": "Mantenimiento_EquipoRelacionado",

    # Planeación
    "PD01": "Planeacion_CambioProgramacion",
    "PD02": "Planeacion_FaltaInsumo",

    # Logística
    "L01": "Logistica_SinMateriaPrima",
    "L02": "Logistica_SinEspacio",

    # Producción
    "P01": "Produccion_ParoOperativo",
    "P02": "Produccion_Convencional",
    "P03": "Produccion_Limpieza",
    "P04": "Produccion_CambioDestino",
    "P05": "Produccion_SinSistemaWMS",
    "P06": "Produccion_Capacitacion",
    "P07": "Produccion_SinLicor",
    "P08": "Produccion_NivelLlenado",

    # Calidad
    "CO1": "Calidad_Embalaje",
    "CO2": "Calidad_Botella",
    "CO3": "Calidad_Tapa",
    "CO4": "Calidad_Etiqueta",
    "CO5": "Calidad_ParametrosFueraRango",

    # Especiales
    "-": "Otros",
    "001": "Falla_Comunicacion",
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
    q = q[q["linea"] == "linea03"].reset_index(drop=True)
    d = d[d["linea"] == "linea03"].reset_index(drop=True)

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
