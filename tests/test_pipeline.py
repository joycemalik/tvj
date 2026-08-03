import os
import pytest
from src.pipeline import run_single_spectrum_pipeline

def test_pipeline_on_example_spectrum():
    spec_path = "70dcdr2dswp35476mxlo.txt"
    if os.path.exists(spec_path):
        record = run_single_spectrum_pipeline(spec_path)
        assert 'amplitude' in record
        assert 'center' in record
        assert 'sigma' in record
        assert record['amplitude'] > 0
        assert 1210.0 <= record['center'] <= 1220.0
