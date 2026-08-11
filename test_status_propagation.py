import pytest
import numpy as np
from src.pipeline import run_single_spectrum_pipeline
import os

def test_status_propagation():
    """
    Test that if a line has high SNR initially, but joint fitting reduces
    its SNR below 3, its fit_status is properly downgraded to NOT_DETECTED.
    We test this on spectrum 18 (18dcdr2dswp23444mxlo.txt), where He II 
    and O III] ended up with SNR < 3 but were originally reported as ACCEPTED.
    """
    # Verify the file exists
    spec_path = '18dcdr2dswp23444mxlo.txt'
    if not os.path.exists(spec_path):
        pytest.skip(f"Test data {spec_path} not found")
        
    res = run_single_spectrum_pipeline(spec_path)
    lines = res.get('lines', [])
    
    for l in lines:
        name = l.get('line_name', '')
        snr = l.get('snr', 0.0)
        status = l.get('fit_status', '')
        
        # If SNR is under 3, it MUST NOT be ACCEPTED
        if snr < 3.0:
            assert 'ACCEPTED' not in status, f"{name} has SNR {snr:.2f} but status is {status}"
            assert 'NOT_DETECTED' in status or 'REJECTED' in status or 'MARGINAL' in status, f"{name} status should be rejected/not detected"
