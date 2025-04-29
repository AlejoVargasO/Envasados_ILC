"""
src/config.py
Carga configuración desde config.yaml y expone getters para pipeline y auth.
"""
import yaml
from pathlib import Path

_cfg = None

def load_config():
    global _cfg
    if _cfg is None:
        # Ubica config.yaml en la raíz del proyecto
        config_path = Path(__file__).parent.parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            _cfg = yaml.safe_load(f)
    return _cfg

def get_pipeline_config():
    cfg = load_config()
    if 'pipeline' not in cfg:
        raise KeyError("Sección 'pipeline' no definida en config.yaml")
    return cfg['pipeline']

def get_auth_config():
    cfg = load_config()
    if 'auth' not in cfg:
        raise KeyError("Sección 'auth' no definida en config.yaml")
    return cfg['auth']
