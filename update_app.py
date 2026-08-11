import sys
import os

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Add imports
if 'import uuid' not in code:
    code = code.replace('import os', 'import os\nimport uuid\nimport threading\nimport queue\nimport contextlib\nimport json\nfrom flask import Response')

tasks_dict = '''
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
                    
                    clean = {}
                    for k, v in record.items():
                        if isinstance(v, np.ndarray):
                            clean[k] = [round(float(x), 6) for x in v]
                        elif k == 'lines':
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
                                if 'components' in line and isinstance(line['components'], list):
                                    for c in line['components']:
                                        flat_comp = dict(line)
                                        flat_comp.update(c)
                                        flat_comp['parent'] = line.get('line_name', '')
                                        flat_comp['component'] = c.get('name', '')
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
            yield f"data: {json.dumps(msg)}\\n\\n"
            
    return Response(generate(), mimetype='text/event-stream')
'''

code = code.replace("def index():\n    return render_template('index.html')", "def index():\n    return render_template('index.html')\n\n" + tasks_dict)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('Updated app.py')
