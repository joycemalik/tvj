import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List

def load_manual_excel_fits(excel_path: str) -> pd.DataFrame:
    """
    Parses Ly.xlsx containing manual fits.
    Header is at row index 1:
      ['spectrum no:', 'AmplitudeE-13', 'Line Center (Å)', 'Wing Window (Å)', 'Sigma', 'Sigma Min', 'Sigma Max']
    """
    df = pd.read_excel(excel_path)
    # The true header row is row 1
    header_row = df.iloc[1].tolist()
    data_df = df.iloc[2:].copy()
    data_df.columns = header_row
    
    # Drop rows where spectrum no is NaN
    data_df = data_df.dropna(subset=['spectrum no:'])
    
    # Rename columns to standard names
    col_map = {
        'spectrum no:': 'spectrum_no',
        'AmplitudeE-13': 'amplitude_e13',
        'Line Center ()': 'center',
        'Line Center (Å)': 'center',
        'Wing Window ()': 'wing_window',
        'Wing Window (Å)': 'wing_window',
        'Sigma': 'sigma',
        'Sigma Min': 'sigma_min',
        'Sigma Max': 'sigma_max'
    }
    
    renamed = {}
    for c in data_df.columns:
        c_str = str(c)
        if c_str in col_map:
            renamed[c] = col_map[c_str]
        elif 'Line Center' in c_str:
            renamed[c] = 'center'
        elif 'Wing Window' in c_str:
            renamed[c] = 'wing_window'
        else:
            renamed[c] = c_str
            
    data_df = data_df.rename(columns=renamed)
    
    # Convert numerical columns
    for col in ['spectrum_no', 'amplitude_e13', 'center', 'wing_window', 'sigma', 'sigma_min', 'sigma_max']:
        if col in data_df.columns:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
            
    data_df['amplitude'] = data_df['amplitude_e13'] * 1.0e-13
    return data_df.dropna(subset=['spectrum_no'])

def compare_single_fit(auto_record: Dict[str, Any], manual_row: pd.Series) -> Dict[str, Any]:
    """
    Compares automatic fit parameters with manual Excel fit row.
    """
    manual_amp = float(manual_row['amplitude'])
    auto_amp = float(auto_record['amplitude'])
    amp_diff_pct = abs(auto_amp - manual_amp) / manual_amp * 100.0 if manual_amp > 0 else 0.0
    
    manual_center = float(manual_row['center'])
    auto_center = float(auto_record['center'])
    center_diff_ang = abs(auto_center - manual_center)
    
    manual_sigma = float(manual_row['sigma'])
    auto_sigma = float(auto_record['sigma'])
    sigma_diff_ang = abs(auto_sigma - manual_sigma)
    
    manual_wing = float(manual_row['wing_window'])
    auto_wing = float(auto_record['wing_window'])
    wing_diff_ang = abs(auto_wing - manual_wing)
    
    # Pass criteria: Amp diff <= 15%, Center diff <= 1.0 Å, Sigma diff <= 1.5 Å
    passed = (amp_diff_pct <= 15.0) and (center_diff_ang <= 1.0) and (sigma_diff_ang <= 1.5)
    
    return {
        'spectrum_no': int(manual_row['spectrum_no']),
        'passed': bool(passed),
        'manual_amp': manual_amp,
        'auto_amp': auto_amp,
        'amp_diff_pct': amp_diff_pct,
        'manual_center': manual_center,
        'auto_center': auto_center,
        'center_diff_ang': center_diff_ang,
        'manual_sigma': manual_sigma,
        'auto_sigma': auto_sigma,
        'sigma_diff_ang': sigma_diff_ang,
        'manual_wing': manual_wing,
        'auto_wing': auto_wing,
        'reduced_chi2': auto_record.get('reduced_chi2', 0.0),
        'snr': auto_record.get('snr', 0.0)
    }
