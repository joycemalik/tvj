import os
import json
import pandas as pd
from typing import List, Dict, Any

def save_batch_results(results: List[Dict[str, Any]], output_dir: str = "outputs"):
    """
    Saves pipeline results across multiple spectra into structured outputs:
    1. CSV summary table (outputs/csv/summary.csv)
    2. JSON metadata per spectrum (outputs/json/<spectrum_name>.json)
    3. Markdown summary report (outputs/reports/summary_report.md)
    """
    csv_dir = os.path.join(output_dir, "csv")
    json_dir = os.path.join(output_dir, "json")
    report_dir = os.path.join(output_dir, "reports")
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    clean_records = []
    for r in results:
        # Exclude large array data from summary records
        r_clean = {}
        for k, v in r.items():
            if k not in ['wavelength', 'observed_flux', 'continuum_fit', 'subtracted_y']:
                if hasattr(v, 'item'):
                    r_clean[k] = v.item()
                else:
                    r_clean[k] = v
        clean_records.append(r_clean)
        
        # Save individual JSON
        spec_name = os.path.splitext(r['spectrum_name'])[0]
        json_path = os.path.join(json_dir, f"{spec_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(r_clean, f, indent=2)
            
    # Save CSV
    df = pd.DataFrame(clean_records)
    csv_path = os.path.join(csv_dir, "emission_line_fits.csv")
    df.to_csv(csv_path, index=False)
    
    # Save Markdown Summary
    md_path = os.path.join(report_dir, "batch_fit_summary.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Batch Emission Line Fitting Summary Report\n\n")
        f.write(f"Total spectra processed: **{len(results)}**\n\n")
        
        accepted_cnt = sum(1 for r in clean_records if r.get('is_acceptable', False))
        f.write(f"- Accepted fits: **{accepted_cnt}** ({accepted_cnt/len(results)*100:.1f}%)\n")
        f.write(f"- Flagged/Rejected fits: **{len(results) - accepted_cnt}**\n\n")
        
        f.write("## Detailed Results Table\n\n")
        headers = ["Spectrum", "Amplitude", "Center (Å)", "Sigma (Å)", "Flux", "FWHM (km/s)", "SNR", "χ²_red", "Status"]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        
        for r in clean_records:
            row = [
                r['spectrum_name'],
                f"{r['amplitude']:.3e}",
                f"{r['center']:.2f}",
                f"{r['sigma']:.2f}",
                f"{r['flux']:.3e}",
                f"{r['fwhm_kms']:.1f}",
                f"{r['snr']:.2f}",
                f"{r['reduced_chi2']:.2f}",
                r['fit_status']
            ]
            f.write("| " + " | ".join(row) + " |\n")
            
    return csv_path, md_path
