"""
src/train.py
Entrena un modelo MLP baseline para pronóstico de velocidad de producción.
- Carga dataset final para la línea configurada
- Separa features (incluye device_idx) y target
- Escala features numéricas
- Guarda feature names y scaler para consistencia en predict.py
- Entrena MLP con EarlyStopping
- Guarda modelo con timestamp
"""
import pandas as pd
from pathlib import Path
import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from config import get_pipeline_config

# ---------------------------------------
# Configuración dinámica
# ---------------------------------------
cfg = get_pipeline_config()
line = cfg['line']

# ---------------------------------------
# Rutas
# ---------------------------------------
PROC_FINAL_DIR = Path('data/processed/final')
MODELS_DIR     = Path('models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# Función para obtener archivo más reciente
# ---------------------------------------
def latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No se encontró archivo con patrón {pattern} en {directory}")
    return files[-1]

# ---------------------------------------
# Entrenamiento
# ---------------------------------------
def train():
    # 1. Cargar datos
    pattern = f"dataset_final_*.parquet"
    data_path = latest_file(PROC_FINAL_DIR, pattern)
    print(f"[TRAIN] Cargando dataset: {data_path.name}")
    df = pd.read_parquet(data_path)

    # 2. Separar features y target
    y = df['velocity_bpm']
    X = df.drop(columns=['_time', 'linea', 'velocity_bpm', 'shift'])
    # Mantener sólo numéricas y device_idx
    X = X.select_dtypes(include=['number'])

    # 3. División train/valid (sin shuffle para señales temporales)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # 4. Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    # Guardar scaler y feature names
    today = datetime.date.today().isoformat()
    scaler_path = MODELS_DIR / f"scaler_{today}.pkl"
    feature_names_path = MODELS_DIR / f"feature_names_{today}.pkl"
    joblib.dump(scaler, scaler_path)
    joblib.dump(X.columns.tolist(), feature_names_path)
    print(f"[TRAIN] Scaler guardado en: {scaler_path}")
    print(f"[TRAIN] Feature names guardados en: {feature_names_path}")

    # 5. Definir modelo MLP
    n_features = X_train_scaled.shape[1]
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 6. Entrenar con EarlyStopping
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=50,
        batch_size=256,
        callbacks=[es],
        verbose=1
    )

    # 7. Guardar modelo
    model_path = MODELS_DIR / f"model_{today}.h5"
    model.save(model_path)
    print(f"[TRAIN] Modelo entrenado guardado en: {model_path}")

if __name__ == '__main__':
    train()
