
# src/ingest.py
# -------------------------------
# Copia los CSV de calidad y disponibilidad desde el servidor local
# a la carpeta data/raw/ con nombre con la fecha actual.
# -------------------------------

import os
import shutil
import glob
import datetime

# ---------------------------------------------
# CONFIGURACIÓN DE RUTAS
# ---------------------------------------------
# Carpeta compartida en red donde el PLC deja los CSV
# Ajusta la IP y ruta según tu servidor
SERVER_DIR = r"\\192.168.1.100\DatosPLC"

# Carpeta local donde guardaremos los CSV descargados
git_root = os.getcwd()  # asume que corres desde la raíz del proyecto
data_raw = os.path.join(git_root, "data", "raw")

# Prefijos de archivos a copiar (calidad y disponibilidad)
PREFIXES = ["calidad", "disponibilidad"]

# ---------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------
def latest_file(prefix: str) -> str:
    """
    Busca en SERVER_DIR el CSV que comience con `prefix` y devuelve el más reciente.
    """
    pattern = os.path.join(SERVER_DIR, f"{prefix}*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No se encontró ningún archivo con prefijo '{prefix}' en {SERVER_DIR}")
    # Ordenar por fecha de modificación y retornar el más reciente
    files.sort(key=os.path.getmtime)
    return files[-1]

# ---------------------------------------------
# PROCESO PRINCIPAL
# ---------------------------------------------
def main():
    # Crear carpeta local si no existe
    os.makedirs(data_raw, exist_ok=True)

    # Para cada prefijo, copiar el último CSV
    today = datetime.date.today().isoformat()
    for prefix in PREFIXES:
        src = latest_file(prefix)
        dst_name = f"{prefix}_{today}.csv"
        dst = os.path.join(data_raw, dst_name)

        shutil.copy2(src, dst)
        print(f"[INGEST] Copiado: {src}\n       -> {dst}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR INGEST] {e}")
        exit(1)
