"""
Entrenamiento del modelo predictivo
Lee el dataset final con features, crea X/y para un paso a futuro,
ajusta un modelo neuronal feed‑forward, y guarda el modelo y scaler.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --------------------------------------
# Rutas y parámetros
device_root = Path.cwd()
data_dir = device_root / "data" / "processed" / "final"
models_dir = device_root / "models"
models_dir.mkdir(exist_ok=True)

# Prefijo del dataset final
FINAL_PATTERN = "dataset_final_*.parquet"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --------------------------------------
# Funciones auxiliares
def latest_final() -> Path:
    files = sorted(data_dir.glob(FINAL_PATTERN), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No hay archivos {FINAL_PATTERN} en {data_dir}")
    return files[-1]

# --------------------------------------
# Pipeline de entrenamiento
def train():
    # 1. Cargar dataset
    final_path = latest_final()
    print(f"[TRAIN] Cargando {final_path.name}")
    df = pd.read_parquet(final_path)

    # 2. Crear y (X, y): predecir velocity_bpm un paso adelante
    df['target'] = df['velocity_bpm'].shift(-1)
    df = df.dropna().reset_index(drop=True)

    # 3. Definir X e y
    feature_cols = [c for c in df.columns if c not in ['_time','velocity_bpm','target','device_id']]
    X = df[feature_cols].values
    y = df['target'].values.reshape(-1,1)

    # 4. División train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
    )

    # 5. Escalado
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Guardar scaler
    scaler_path = models_dir / f"scaler_{datetime.date.today()}.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"[TRAIN] Scaler guardado en: {scaler_path}")

    # 6. Definir modelo
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 7. Callbacks
    checkpoint_path = models_dir / f"model_{datetime.date.today()}.h5"
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
    ]

    # 8. Entrenamiento
    print(f"[TRAIN] Iniciando fit, X_train shape={X_train.shape}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )

    print(f"[TRAIN] Modelo entrenado guardado en: {checkpoint_path}")

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        print(f"[ERROR TRAIN] {e}")
        raise
