import os
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from src.pipeline import run_single_spectrum_pipeline
from src.plotting import save_publication_plots

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
PLOT_FOLDER   = os.path.join(os.path.dirname(__file__), 'outputs', 'plots')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER,   exist_ok=True)

DEMO_FILE = os.path.join(os.path.dirname(__file__), '70dcdr2dswp35476mxlo.txt')


def _serialize(v):
    """Recursively convert numpy scalars and dicts for JSON."""
    if isinstance(v, np.ndarray):
        return None                      # drop arrays
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, dict):
        return {k2: _serialize(v2) for k2, v2 in v.items()}
    if isinstance(v, list):
        return [_serialize(item) for item in v]
    return v


@app.route('/')
def index():
    return render_template('index.html')


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

        # Serialize – drop numpy arrays, keep scalars + shortlist
        clean = {}
        for k, v in record.items():
            serialized = _serialize(v)
            if serialized is not None:
                clean[k] = serialized

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
