import os
import uuid
import threading
import queue
import contextlib
import json
from flask import Response
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from src.pipeline import run_single_spectrum_pipeline
from src.plotting import save_publication_plots

app = Flask(__name__, template_folder='templates')
# Detect Vercel or other serverless env where local storage is read-only
IS_SERVERLESS = os.environ.get('VERCEL') == '1' or not os.access(os.path.dirname(__file__) or '.', os.W_OK)

if IS_SERVERLESS:
    UPLOAD_FOLDER = '/tmp/uploads'
    PLOT_FOLDER   = '/tmp/plots'
else:
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    PLOT_FOLDER   = os.path.join(os.path.dirname(__file__), 'outputs', 'plots')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER,   exist_ok=True)

DEMO_FILE = os.path.join(os.path.dirname(__file__), '70dcdr2dswp35476mxlo.txt')


def _num(x):
    """JSON-safe float: 6 significant figures (fluxes are ~1e-13), NaN/inf -> None."""
    x = float(x)
    return float(f'{x:.6g}') if np.isfinite(x) else None


def _serialize(v):
    """Recursively convert numpy scalars, arrays, dicts and lists for JSON."""
    if isinstance(v, np.ndarray):
        return [_num(x) for x in v.ravel()]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return _num(v)
    if isinstance(v, dict):
        return {k2: _serialize(v2) for k2, v2 in v.items()}
    if isinstance(v, (list, tuple)):
        return [_serialize(item) for item in v]
    return v


def _serialize_record(record):
    """Pipeline record -> JSON-safe dict (arrays kept for plotting)."""
    return {k: _serialize(v) for k, v in record.items()}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/v2')
def index_v2():
    return render_template('index_v2.html')


@app.route('/v3')
def index_v3():
    return render_template('spectrum-analysis.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/api/fvar_context')
def fvar_context():
    """Observation date, same-year light curve and F_var for one spectrum (?spectrum=<filename>)."""
    from src.variability import spectrum_context
    name = request.args.get('spectrum', '')
    if not name:
        return jsonify({'error': 'spectrum parameter required'}), 400
    return jsonify(spectrum_context(name))


@app.route('/api/fvar')
def fvar_table():
    """Per-line, per-year F_var computed by run_variability.py."""
    from src.variability import load_fvar_table, FVAR_ERR_CUTOFF
    rows = load_fvar_table()
    if rows is None:
        return jsonify({'error': 'outputs/fvar_by_year.csv not found — run `python run_variability.py`'}), 404
    return jsonify({'cutoff': FVAR_ERR_CUTOFF, 'lines': list(dict.fromkeys(r['line'] for r in rows)), 'rows': rows})


tasks = {}

class QueueWriter:
    def __init__(self, q):
        self.q = q
    def write(self, msg):
        if msg.strip():
            self.q.put({'type': 'log', 'message': msg.strip()})
    def flush(self):
        pass

@app.route('/upload_fit', methods=['POST'])
def upload_fit():
    is_demo = request.form.get('demo') == 'true'
    task_id = str(uuid.uuid4())
    
    if is_demo:
        filepath = DEMO_FILE
    else:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        if file and file.filename.endswith('.txt'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
        else:
            return jsonify({'error': 'Invalid file type. Please upload a .txt spectrum file.'}), 400
            
    tasks[task_id] = {'filepath': filepath, 'queue': queue.Queue(), 'is_demo': is_demo}
    return jsonify({'task_id': task_id})

@app.route('/stream/<task_id>')
def stream(task_id):
    if task_id not in tasks:
        return "Not found", 404
        
    task = tasks[task_id]
    
    def generate():
        q = task['queue']
        
        def run_pipeline():
            qw = QueueWriter(q)
            with contextlib.redirect_stdout(qw):
                try:
                    print('[SYSTEM] Starting Scientific Diagnostic Fitting Pipeline...')
                    record = run_single_spectrum_pipeline(task['filepath'])
                    print('[SYSTEM] Processing completed. Generating plots...')
                    plot_path = save_publication_plots(record, output_dir=PLOT_FOLDER)
                    plot_filename = os.path.basename(plot_path)
                    
                    clean = _serialize_record(record)
                    clean['plot_url'] = f'/plots/{plot_filename}'
                    
                    if not task['is_demo']:
                        try: os.remove(task['filepath'])
                        except Exception: pass
                        
                    q.put({'type': 'result', 'data': clean})
                except Exception as e:
                    import traceback
                    if not task['is_demo']:
                        try: os.remove(task['filepath'])
                        except Exception: pass
                    q.put({'type': 'error', 'message': str(e), 'trace': traceback.format_exc()})
            q.put(None)
            
        t = threading.Thread(target=run_pipeline)
        t.start()
        
        while True:
            msg = q.get()
            if msg is None:
                break
            yield f"data: {json.dumps(msg)}\n\n"
            
    return Response(generate(), mimetype='text/event-stream')



@app.route('/fit', methods=['POST'])
def fit_spectrum():
    is_demo = request.form.get('demo') == 'true'

    if is_demo:
        filepath = DEMO_FILE
        filename = os.path.basename(DEMO_FILE)
    else:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        if file and file.filename.endswith('.txt'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
        else:
            return jsonify({'error': 'Invalid file type. Please upload a .txt spectrum file.'}), 400

    try:
        record    = run_single_spectrum_pipeline(filepath)
        plot_path = save_publication_plots(record, output_dir=PLOT_FOLDER)
        plot_filename = os.path.basename(plot_path)

        clean = _serialize_record(record)
        clean['plot_url'] = f'/plots/{plot_filename}'

        if not is_demo:
            try:
                os.remove(filepath)
            except Exception:
                pass

        return jsonify(clean)

    except Exception as e:
        if not is_demo:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                pass
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/plots/<filename>')
def get_plot(filename):
    return send_from_directory(PLOT_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
