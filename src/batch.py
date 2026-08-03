import os
import glob
from typing import List, Dict, Any, Optional

from src.pipeline import run_single_spectrum_pipeline
from src.plotting import save_publication_plots
from src.report import save_batch_results

def run_batch_pipeline(
    spectra_dir: str,
    line_config_path: str = "config/Lya.yaml",
    output_dir: str = "outputs",
    generate_plots: bool = True
) -> List[Dict[str, Any]]:
    """
    Processes all text spectrum files in `spectra_dir` in batch,
    saving plots, CSVs, JSON, and summary reports.
    """
    pattern = os.path.join(spectra_dir, "*.txt")
    spectrum_files = glob.glob(pattern)
    
    if not spectrum_files:
        print(f"No .txt spectrum files found in directory: {spectra_dir}")
        return []
        
    print(f"Found {len(spectrum_files)} spectra for batch processing...")
    results = []
    
    for i, filepath in enumerate(spectrum_files, 1):
        filename = os.path.basename(filepath)
        print(f"[{i}/{len(spectrum_files)}] Processing {filename}...")
        try:
            record = run_single_spectrum_pipeline(filepath, line_config_path=line_config_path)
            if generate_plots:
                save_publication_plots(record, output_dir=os.path.join(output_dir, "plots"))
            results.append(record)
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            
    if results:
        csv_path, md_path = save_batch_results(results, output_dir=output_dir)
        print(f"\nBatch processing complete! Summary saved to:\n  - CSV: {csv_path}\n  - Markdown: {md_path}")
        
    return results
