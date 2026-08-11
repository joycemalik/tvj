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

candidates = generate_candidates(wl, sub_y, cont, 1216.0, config, top_n=5)
for c in candidates:
    print(f"score={c['composite_score']:.4f} wing={c['wing_window']} sigma={c['sigma']:.3f} center={c['center']:.3f}")
