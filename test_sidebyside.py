import numpy as np
from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.candidate_engine import generate_candidates
import yaml
from src.statistics import estimate_noise

wl, flux = load_spectrum('15dcdr2dswp19731mxlo.txt')
_, _, cont, sub_y = fit_continuum(wl, flux, '1150:1175,1300:1350')
mean_cont = float(np.mean(cont))
norm_res = sub_y / abs(mean_cont)
noise = estimate_noise(norm_res) * abs(mean_cont)
print(f"NOISE test_weights: {noise}")

with open('config/Lya.yaml', 'r') as f:
    config = yaml.safe_load(f)

# candidate_engine uses:
# candidates = generate_candidates(wl, sub_y, cont, 1216.0, config, top_n=5)
# Let's print noise from candidate_engine:
def mock_generate(*args, **kwargs):
    print("INSIDE MOCK!")
