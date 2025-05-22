"""
app.py: API Flask + Scheduler para ingestión, merge, entrenamiento y pronóstico dinámico
- Autenticación básica HTTP desde config.yaml
- Logging de solicitudes y métricas en app.log
- Endpoints protegidos: /, /ingest, /merge, /train, /forecast, /forecast/data, /metrics
- Configuración dinámica de línea y horas para forecast
"""
import sys
import os
import datetime
import subprocess
import logging
from functools import wraps
from flask import Flask, jsonify, render_template, send_file, request, Response, g
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
from src.config import get_pipeline_config, get_auth_config

# ----------------------------------
# Logging
# ----------------------------------
LOG_FILE = 'app.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
START_TIME = datetime.datetime.now()
REQUEST_COUNT = 0

# ----------------------------------
# Flask app & scheduler
# ----------------------------------
app = Flask(__name__)
scheduler = BackgroundScheduler()

# ----------------------------------
# Auth
# ----------------------------------
auth_cfg = get_auth_config()
USERNAME = auth_cfg.get('user')
PASSWORD = auth_cfg.get('pass')

def check_auth(user, pwd): return user == USERNAME and pwd == PASSWORD
def authenticate(): return Response('Authentication required', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# ----------------------------------
# Paths & commands
# ----------------------------------
cfg = get_pipeline_config()
BASE_DIR   = os.getcwd()
SRC_DIR    = os.path.join(BASE_DIR, 'src')
DATA_DIR   = os.path.join(BASE_DIR, 'data', 'predictions')
PYTHON_EXE = sys.executable

CMD_INGEST = [PYTHON_EXE, os.path.join(SRC_DIR, 'ingest.py')]
CMD_MERGE  = [PYTHON_EXE, os.path.join(SRC_DIR, 'merge_quality_availability.py')]
CMD_TRAIN  = [PYTHON_EXE, os.path.join(SRC_DIR, 'train.py')]
# builder for predict

def build_predict_cmd(line, hours):
    return [PYTHON_EXE, os.path.join(SRC_DIR, 'predict.py'), '--line', line, '--hours', str(hours)]

# ----------------------------------
# Helper to run scripts
# ----------------------------------
def run_script(cmd, name):
    try:
        logger.info(f"Running {name}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info(f"{name} completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {name}: {e}")

# ----------------------------------
# Request logging
# ----------------------------------
@app.before_request
def before_request():
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    g.start = datetime.datetime.now()
    logger.info(f"Request {REQUEST_COUNT}: {request.method} {request.path}")

@app.after_request
def after_request(response):
    delta = datetime.datetime.now() - g.start
    logger.info(f"Completed {request.method} {request.path} in {delta.total_seconds():.3f}s -> {response.status_code}")
    return response

# ----------------------------------
# Endpoints
# ----------------------------------
@app.route('/', methods=['GET'])
@requires_auth
def index():
    return render_template('index.html')

@app.route('/ingest', methods=['GET'])
@requires_auth
def ingest():
    run_script(CMD_INGEST, 'ingest')
    return jsonify(status='ingest completed', timestamp=str(datetime.datetime.now()))

@app.route('/merge', methods=['GET'])
@requires_auth
def merge():
    run_script(CMD_MERGE, 'merge')
    return jsonify(status='merge completed', timestamp=str(datetime.datetime.now()))

@app.route('/train', methods=['GET'])
@requires_auth
def train():
    run_script(CMD_TRAIN, 'train')
    return jsonify(status='train completed', timestamp=str(datetime.datetime.now()))

@app.route('/forecast', methods=['GET'])
@requires_auth
def forecast_csv():
    line = request.args.get('line', cfg['line'])
    hours = int(request.args.get('hours', cfg['horizon_hours']))
    cmd = build_predict_cmd(line, hours)
    run_script(cmd, 'predict')
    today = datetime.date.today().isoformat()
    fname = f'forecast_{line}_{hours}h_{today}.csv'
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        return send_file(fpath, mimetype='text/csv', as_attachment=True)
    return jsonify(error='file not found', path=fpath), 404

@app.route('/forecast/data', methods=['GET'])
@requires_auth
def forecast_data():
    line = request.args.get('line', cfg['line'])
    hours = int(request.args.get('hours', cfg['horizon_hours']))
    cmd = build_predict_cmd(line, hours)
    run_script(cmd, 'predict')
    today = datetime.date.today().isoformat()
    fname = f'forecast_{line}_{hours}h_{today}.csv'
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        return jsonify(error='file not found', path=fpath), 404
    df = pd.read_csv(fpath, parse_dates=['_time'])
    data = df.to_dict(orient='records')
    return jsonify(data)

@app.route('/metrics', methods=['GET'])
@requires_auth
def metrics():
    uptime = datetime.datetime.now() - START_TIME
    return jsonify(uptime_seconds=uptime.total_seconds(), request_count=REQUEST_COUNT, start=START_TIME.isoformat())

# ----------------------------------
# Scheduler
# ----------------------------------
scheduler.add_job(lambda: run_script(CMD_INGEST, 'ingest'),  'cron', hour=cfg['ingest_hour_morning'], minute=cfg['ingest_minute_morning'])
scheduler.add_job(lambda: run_script(CMD_MERGE, 'merge'),    'cron', hour=cfg['merge_hour_morning'],  minute=cfg['merge_minute_morning'])
scheduler.add_job(lambda: run_script(CMD_TRAIN, 'train'),    'cron', hour=cfg['train_hour_morning'],  minute=cfg['train_minute_morning'])
scheduler.add_job(lambda: run_script(build_predict_cmd(cfg['line'], cfg['horizon_hours']), 'predict'), 'cron', hour=cfg['forecast_hour_morning'], minute=cfg['forecast_minute_morning'])

scheduler.add_job(lambda: run_script(CMD_INGEST, 'ingest'),  'cron', hour=cfg['ingest_hour_evening'], minute=cfg['ingest_minute_evening'])
scheduler.add_job(lambda: run_script(CMD_MERGE, 'merge'),    'cron', hour=cfg['merge_hour_evening'],  minute=cfg['merge_minute_evening'])
scheduler.add_job(lambda: run_script(CMD_TRAIN, 'train'),    'cron', hour=cfg['train_hour_evening'],  minute=cfg['train_minute_evening'])
scheduler.add_job(lambda: run_script(build_predict_cmd(cfg['line'], cfg['horizon_hours']), 'predict'), 'cron', hour=cfg['forecast_hour_evening'], minute=cfg['forecast_minute_evening'])

scheduler.start()

# ----------------------------------
# Main
# ----------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
