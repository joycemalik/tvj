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


from src.serialize import serialize_record as _serialize_record  # noqa: E402


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/v2')
def index_v2():
    return render_template('index_v2.html')


@app.route('/v3')
def index_v3():
    return render_template('spectrum-analysis.html')


@app.route('/v4')
def index_v4():
    return render_template('manuscript.html')


@app.route('/v5')
def index_v5():
    return render_template('campaign.html')


# ---- Campaign (all spectra) -------------------------------------------------
CAMPAIGN_JOBS = {}          # job id -> {'done', 'total', 'state', 'error'}
_ROOT = os.path.dirname(os.path.abspath(__file__))


def _campaign_dir(job):
    """Result folder: precomputed campaign (job '' ) or an uploaded batch."""
    if not job:
        return os.path.join(_ROOT, 'outputs')
    if not all(c.isalnum() for c in job):
        return None
    return os.path.join(_ROOT, 'outputs', 'batches', job)


@app.route('/api/campaign/summary')
def campaign_summary():
    """All fitted lines of all spectra (from line_fluxes.csv) for the results table."""
    import csv
    d = _campaign_dir(request.args.get('job', ''))
    path = d and os.path.join(d, 'line_fluxes.csv')
    if not path or not os.path.exists(path):
        return jsonify({'error': 'no campaign results yet — run `python run_variability.py` or process spectra'}), 404
    keep = ('spectrum', 'line', 'rest_wavelength', 'detected', 'verification', 'failed_checks', 'center', 'center_err',
            'sigma', 'sigma_err', 'fwhm_kms', 'fwhm_kms_intrinsic', 'flux', 'flux_err', 'significance', 'ew', 'ew_err',
            'reduced_chi2', 'wing_window', 'min_wavelength', 'max_wavelength', 'spectral_index', 'spectral_index_err',
            'continuum_amplitude', 'continuum_amplitude_err', 'jd', 'date', 'year')
    rows = []
    with open(path, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            out = {}
            for k in keep:
                v = r.get(k, '')
                if k in ('spectrum', 'line', 'verification', 'failed_checks', 'date'):
                    out[k] = v[:10] if k == 'date' else v
                elif k == 'detected':
                    out[k] = v == 'True'
                else:
                    try:
                        x = float(v)
                        out[k] = x if np.isfinite(x) else None
                    except ValueError:
                        out[k] = None
            rows.append(out)
    has_json = os.path.isdir(os.path.join(d, 'campaign'))
    return jsonify({'rows': rows, 'has_spectra': has_json,
                    'workbook': '/download/campaign.xlsx' + (f'?job={request.args.get("job")}' if request.args.get('job') else '')})


@app.route('/api/campaign/spectrum/<path:name>')
def campaign_spectrum(name):
    d = _campaign_dir(request.args.get('job', ''))
    if d is None:
        return jsonify({'error': 'bad job'}), 400
    # send_from_directory refuses paths that escape the folder
    return send_from_directory(os.path.join(d, 'campaign'), secure_filename(name) + '.json', mimetype='application/json')


@app.route('/download/campaign.xlsx')
def campaign_workbook():
    d = _campaign_dir(request.args.get('job', ''))
    if d is None or not os.path.exists(os.path.join(d, 'variability_results.xlsx')):
        return jsonify({'error': 'no workbook yet'}), 404
    return send_from_directory(d, 'variability_results.xlsx', as_attachment=True,
                               download_name='3C273_campaign_results.xlsx')


@app.route('/api/campaign/process', methods=['POST'])
def campaign_process():
    """Fit many uploaded .txt spectra in parallel (background thread + worker processes)."""
    files = [f for f in request.files.getlist('files') if f.filename.lower().endswith('.txt')]
    if not files:
        return jsonify({'error': 'upload one or more .txt spectra'}), 400
    job = uuid.uuid4().hex[:12]
    d = _campaign_dir(job)
    src = os.path.join(d, 'input')
    os.makedirs(src, exist_ok=True)
    paths = []
    for f in files:
        p = os.path.join(src, secure_filename(f.filename))
        f.save(p)
        paths.append(p)
    CAMPAIGN_JOBS[job] = {'done': 0, 'total': len(paths), 'state': 'running', 'error': None}

    def work():
        from src.campaign import run_batch
        try:
            run_batch(paths, d, 'refine', None, os.path.join(_ROOT, 'jd.xlsx'), os.path.join(d, 'campaign'),
                      progress=lambda i, n: CAMPAIGN_JOBS[job].update(done=i))
            CAMPAIGN_JOBS[job]['state'] = 'done'
        except Exception as e:
            import traceback
            app.logger.error(traceback.format_exc())
            CAMPAIGN_JOBS[job].update(state='error', error=f'{type(e).__name__}: {e}')

    threading.Thread(target=work, daemon=True).start()
    return jsonify({'job': job, 'total': len(paths)})


@app.route('/api/campaign/status/<job>')
def campaign_status(job):
    s = CAMPAIGN_JOBS.get(job)
    return (jsonify(s), 200) if s else (jsonify({'error': 'unknown job'}), 404)


@app.route('/about')
def about():
    return render_template('about.html')


METHODS = ('refine', 'manuscript')


def _method():
    m = request.values.get('method', 'refine')
    return m if m in METHODS else 'refine'


def _outputs_dir(method):
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    return base if method == 'refine' else os.path.join(base, 'manuscript')


@app.route('/download/variability_results.xlsx')
def download_variability_workbook():
    """Organised campaign workbook written by run_variability.py (?method=manuscript for the manuscript method)."""
    method = _method()
    out = _outputs_dir(method)
    if not os.path.exists(os.path.join(out, 'variability_results.xlsx')):
        return jsonify({'error': f'run `python run_variability.py --method {method}` first'}), 404
    name = '3C273_variability_results.xlsx' if method == 'refine' else '3C273_variability_results_manuscript.xlsx'
    return send_from_directory(out, 'variability_results.xlsx', as_attachment=True, download_name=name)


@app.route('/api/fvar_context')
def fvar_context():
    """Observation date, same-year light curve and F_var for one spectrum (?spectrum=<filename>)."""
    from src.variability import spectrum_context
    name = request.args.get('spectrum', '')
    if not name:
        return jsonify({'error': 'spectrum parameter required'}), 400
    return jsonify(spectrum_context(name, _outputs_dir(_method())))


@app.route('/api/fvar')
def fvar_table():
    """Per-line, per-year F_var computed by run_variability.py."""
    from src.variability import load_fvar_table, FVAR_ERR_CUTOFF
    method = _method()
    rows = load_fvar_table(os.path.join(_outputs_dir(method), 'fvar_by_year.csv'))
    if rows is None:
        return jsonify({'error': f'fvar_by_year.csv not found — run `python run_variability.py --method {method}`'}), 404
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
        method    = _method()
        record    = run_single_spectrum_pipeline(filepath, method=method)
        sub       = '' if method == 'refine' else method
        plot_path = save_publication_plots(record, output_dir=os.path.join(PLOT_FOLDER, sub) if sub else PLOT_FOLDER)
        plot_filename = os.path.basename(plot_path)

        clean = _serialize_record(record)
        clean['plot_url'] = f'/plots/{sub + "/" if sub else ""}{plot_filename}'

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


@app.route('/plots/<path:filename>')
def get_plot(filename):
    # send_from_directory rejects paths escaping PLOT_FOLDER
    return send_from_directory(PLOT_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
