import os
import pytest
import pandas as pd
from src.validation import load_manual_excel_fits, compare_single_fit

def test_load_manual_excel_fits():
    excel_path = "Ly.xlsx"
    if os.path.exists(excel_path):
        df = load_manual_excel_fits(excel_path)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "spectrum_no" in df.columns
        assert "amplitude" in df.columns
        assert "center" in df.columns
        assert "sigma" in df.columns

def test_compare_single_fit():
    auto_record = {
        'amplitude': 8.5e-13,
        'center': 1215.8,
        'sigma': 6.2,
        'wing_window': 10.0,
        'reduced_chi2': 1.2,
        'snr': 8.5
    }
    manual_row = pd.Series({
        'spectrum_no': 2,
        'amplitude': 8.8e-13,
        'center': 1215.5,
        'sigma': 6.0,
        'wing_window': 10.0
    })
    
    res = compare_single_fit(auto_record, manual_row)
    assert res['spectrum_no'] == 2
    assert res['passed'] is True
    assert pytest.approx(res['amp_diff_pct']) == abs(8.5e-13 - 8.8e-13) / 8.8e-13 * 100.0
    assert pytest.approx(res['center_diff_ang']) == 0.3
    assert pytest.approx(res['sigma_diff_ang']) == 0.2
