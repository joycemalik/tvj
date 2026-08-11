import numpy as np
import yaml
from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.candidate_engine import generate_candidates

wl, flux = load_spectrum('15dcdr2dswp19731mxlo.txt')
_, _, cont, sub_y = fit_continuum(wl, flux, '1150:1175,1300:1350')

with open('config/Lya.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['peak_search_radius'] = 15.0

original_generate = generate_candidates
with open('src/candidate_engine.py', 'r') as f:
    content = f.read()

content = content.replace('if proxy_score > best_proxy_w:', 'if wing == 20.0 and abs(center-1215.0)<1e-3:\n                    print(f"sigma={sigma:.3f} proxy={proxy_score:.3f}")\n                if proxy_score > best_proxy_w:')

with open('src/candidate_engine_patched.py', 'w') as f:
    f.write(content)

import importlib.util
spec = importlib.util.spec_from_file_location("candidate_engine_patched", "src/candidate_engine_patched.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

foo.generate_candidates(wl, sub_y, cont, 1216.0, config, top_n=5)
