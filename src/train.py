"""
src/train.py
Entrena un modelo MLP mejorado para pronóstico de velocidad de producción.
- Carga dataset final para la línea configurada
- Separa features (incluye device_idx) y target
- Escala features numéricas
- Guarda feature names y scaler para consistencia en predict.py
- Entrena MLP con EarlyStopping y mejores hiperparámetros
- Guarda modelo con timestamp
"""
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
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
# Agregar features temporales avanzados
# ---------------------------------------
def add_advanced_time_features(df):
    df['_time'] = pd.to_datetime(df['_time'])
    
    # Features básicos
    df['hour'] = df['_time'].dt.hour
    df['minute'] = df['_time'].dt.minute
    df['dayofweek'] = df['_time'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Features cíclicos
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute']/60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute']/60)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    
    # Turno
    df['shift'] = df['hour'].apply(lambda h: 'day' if 7 <= h < 19 else 'night')
    
    return df

# ---------------------------------------
# Entrenamiento
# ---------------------------------------
def train():
    # 1. Cargar datos
    pattern = f"dataset_final_*.parquet"
    data_path = latest_file(PROC_FINAL_DIR, pattern)
    print(f"[TRAIN] Cargando dataset: {data_path.name}")
    df = pd.read_parquet(data_path)
    
    # 2. Filtrar por línea
    df = df[df['linea'] == line].reset_index(drop=True)
    print(f"[TRAIN] Datos para línea {line}: {len(df)} registros")

    # 3. Agregar features temporales avanzados
    df = add_advanced_time_features(df)

    # 4. Separar features y target
    y = df['velocity_bpm']
    X = df.drop(columns=['_time', 'linea', 'velocity_bpm', 'shift'])
    # Mantener sólo numéricas y device_idx
    X = X.select_dtypes(include=['number'])

    # 5. División train/valid (sin shuffle para señales temporales)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # 6. Escalado
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

    # 7. Definir modelo MLP mejorado
    n_features = X_train_scaled.shape[1]
    model = Sequential([
        Input(shape=(n_features,)),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    # Optimizador con learning rate adaptativo
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # 8. Callbacks
    es = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    # 9. Entrenar con EarlyStopping y ReduceLROnPlateau
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[es, reduce_lr],
        verbose=1
    )

    # 10. Guardar modelo
    model_path = MODELS_DIR / f"model_{today}.h5"
    model.save(model_path)
    print(f"[TRAIN] Modelo entrenado guardado en: {model_path}")
    
    # 11. Imprimir métricas finales
    val_loss = history.history['val_loss'][-1]
    val_mae = history.history['val_mae'][-1]
    print(f"[TRAIN] Métricas finales - Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

if __name__ == '__main__':
    train()
