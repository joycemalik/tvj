import sys
import numpy as np
from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.candidate_engine import generate_candidates
from config import DEFAULT_CONTINUUM_WINDOWS

def main():
    wl, fl = load_spectrum('15dcdr2dswp19731mxlo.txt')
    amp, idx, cont, sub = fit_continuum(wl, fl, DEFAULT_CONTINUUM_WINDOWS)

    cands = generate_candidates(wl, sub, cont, top_n=20)

    print(f"{'Rank':<4} | {'Center':>7} | {'Sigma':>6} | {'Wing':>4} | {'Total J':>7} | {'Core X2':>7} | {'Wing X2':>7} | {'R_prof':>7} | {'Score'}")
    print('-' * 80)
    for i, c in enumerate(cands[:15]):
        comp = c['score_components']
        print(f"{i+1:<4} | {c['center']:>7.2f} | {c['sigma']:>6.2f} | {c['wing_window']:>4} | {comp['J_shape']:>7.2f} | {comp['chi2_core']:>7.2f} | {comp['chi2_wing']:>7.2f} | {comp['R_profile']:>7.4f} | {c['composite_score']:.4f}")

if __name__ == '__main__':
    main()
