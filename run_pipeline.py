#!/usr/bin/env python
"""
Automated Gaussian Emission Line Fitting System for AGN Spectra
Usage:
    python run_pipeline.py --spectrum 70dcdr2dswp35476mxlo.txt
    python run_pipeline.py --batch spectra/
"""

import argparse
import sys
import os

from src.pipeline import run_single_spectrum_pipeline
from src.batch import run_batch_pipeline
from src.plotting import save_publication_plots
from src.report import save_batch_results

def main():
    parser = argparse.ArgumentParser(description="Automated Gaussian Emission Line Fitting Pipeline for AGN Spectra")
    parser.add_argument("--spectrum", type=str, help="Path to a single spectrum .txt file")
    parser.add_argument("--batch", type=str, help="Directory containing multiple spectrum .txt files")
    parser.add_argument("--config", type=str, default="config/Lya.yaml", help="Path to emission line config YAML")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    if args.spectrum:
        print(f"Processing single spectrum: {args.spectrum}")
        record = run_single_spectrum_pipeline(args.spectrum, line_config_path=args.config)
        plot_path = save_publication_plots(record, output_dir=os.path.join(args.output, "plots"))
        
        save_batch_results([record], output_dir=args.output)
        
        print("\n--- FIT RESULTS ---")
        print(f"Spectrum:          {record['spectrum_name']}")
        print(f"Status:            {record['fit_status']}")
        print(f"Amplitude:         {record['amplitude']:.4e}")
        print(f"Line Center:       {record['center']:.2f} Å")
        print(f"Sigma:             {record['sigma']:.2f} Å")
        print(f"Wing Window:       {record['wing_window']:.1f} Å")
        print(f"Reduced Chi2:      {record['reduced_chi2']:.2f}")
        print(f"Flux:              {record['flux']:.4e}")
        print(f"SNR:               {record['snr']:.2f}")
        print(f"FWHM:              {record['fwhm_kms']:.1f} km/s ({record['fwhm_ang']:.2f} Å)")
        print(f"Blue Wing:         {record['blue_wing']:.2f} Å")
        print(f"Red Wing:          {record['red_wing']:.2f} Å")
        print(f"Plot saved to:     {plot_path}")
        
    elif args.batch:
        print(f"Processing batch directory: {args.batch}")
        run_batch_pipeline(args.batch, line_config_path=args.config, output_dir=args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
