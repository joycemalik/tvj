import os
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

        # Serialize – keep numeric arrays for full spectrum plotting in UI
        clean = {}
        for k, v in record.items():
            if isinstance(v, np.ndarray):
                clean[k] = [round(float(x), 6) for x in v]
            elif k == 'lines':
                # Serialize each line result dict individually
                clean_lines = []
                for line in (v or []):
                    def _clean_line(l):
                        cl = {}
                        for lk, lv in l.items():
                            if isinstance(lv, np.ndarray):
                                cl[lk] = [round(float(x), 6) for x in lv]
                            else:
                                ser = _serialize(lv)
                                if ser is not None:
                                    cl[lk] = ser
                        return cl

                    # Flatten hierarchical components for UI graphing
                    if 'components' in line and isinstance(line['components'], list):
                        for c in line['components']:
                            flat_comp = dict(line)  # copy parent attrs
                            flat_comp.update(c)     # overwrite with component attrs
                            flat_comp['parent'] = line.get('line_name', '')
                            flat_comp['component'] = c.get('name', '')
                            # Remove the raw components list to avoid circular/nested issues
                            if 'components' in flat_comp:
                                del flat_comp['components']
                            clean_lines.append(_clean_line(flat_comp))
                    else:
                        clean_lines.append(_clean_line(line))
                clean['lines'] = clean_lines
            else:
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
