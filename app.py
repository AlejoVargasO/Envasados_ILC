"""
app.py: API Flask con APScheduler para ejecutar pipelines de ingestión,
merge, entrenamiento y predicción de producción de la línea.
Endpoints:
  - GET /ingest     -> Ejecuta ingest.py
  - GET /merge      -> Ejecuta merge_quality_availability.py
  - GET /train      -> Ejecuta train.py
  - GET /forecast   -> Ejecuta predict.py y devuelve JSON

Scheduler:
  - Al inicio de cada turno programa las tareas automáticamente.
"""
from flask import Flask, jsonify, send_file
from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import datetime
import os

# Configuración de paths
BASE_DIR = os.getcwd()
SRC_DIR = os.path.join(BASE_DIR, 'src')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'predictions')

# Comandos para ejecutar scripts
CMD_INGEST = ['python', os.path.join(SRC_DIR, 'ingest.py')]
CMD_MERGE  = ['python', os.path.join(SRC_DIR, 'merge_quality_availability.py')]
CMD_TRAIN  = ['python', os.path.join(SRC_DIR, 'train.py')]
CMD_PREDICT= ['python', os.path.join(SRC_DIR, 'predict.py')]

app = Flask(__name__)
scheduler = BackgroundScheduler()

# ----------------------------------
# Funciones para ejecutar scripts
# ----------------------------------
def run_ingest():
    subprocess.run(CMD_INGEST, check=False)

def run_merge():
    subprocess.run(CMD_MERGE, check=False)

def run_train():
    subprocess.run(CMD_TRAIN, check=False)

def run_predict():
    subprocess.run(CMD_PREDICT, check=False)

# ----------------------------------
# Endpoints API
# ----------------------------------
from flask import render_template

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/ingest', methods=['GET'])
def endpoint_ingest():
    run_ingest()
    return jsonify({ 'status': 'ingest completed', 'timestamp': str(datetime.datetime.now()) })

@app.route('/merge', methods=['GET'])
def endpoint_merge():
    run_merge()
    return jsonify({ 'status': 'merge completed', 'timestamp': str(datetime.datetime.now()) })

@app.route('/train', methods=['GET'])
def endpoint_train():
    run_train()
    return jsonify({ 'status': 'train completed', 'timestamp': str(datetime.datetime.now()) })

@app.route('/forecast', methods=['GET'])
def endpoint_forecast():
    run_predict()
    # Asumimos que predict.py genera un CSV con nombre 'prediction_YYYY-MM-DD.csv'
    today = datetime.date.today().isoformat()
    csv_path = os.path.join(DATA_DIR, f'forecast_{today}.csv')
    if os.path.exists(csv_path):
        return send_file(csv_path, mimetype='text/csv')
    else:
        return jsonify({ 'error': 'prediction file not found', 'path': csv_path }), 404

# ----------------------------------
# Programación de tareas (ejemplo)
# ----------------------------------
# Turnos: 7:00 y 16:00
# Ajusta horas y minutos según tu calendario real de turnos.
scheduler.add_job(func=run_ingest, trigger='cron', hour=6, minute=55)
scheduler.add_job(func=run_merge,  trigger='cron', hour=6, minute=58)
scheduler.add_job(func=run_train,  trigger='cron', hour=7, minute=5)
scheduler.add_job(func=run_predict,trigger='cron', hour=7, minute=10)

scheduler.add_job(func=run_ingest, trigger='cron', hour=15, minute=55)
scheduler.add_job(func=run_merge,  trigger='cron', hour=15, minute=58)
scheduler.add_job(func=run_train,  trigger='cron', hour=16, minute=5)
scheduler.add_job(func=run_predict,trigger='cron', hour=16, minute=10)

scheduler.start()

# ----------------------------------
# Main
# ----------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
