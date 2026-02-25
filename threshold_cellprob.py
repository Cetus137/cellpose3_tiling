#!/usr/bin/env python3
"""
threshold_cellprob.py - Threshold cell probability maps

This script reads cell probability maps (files containing 'cellprob' in their name),
applies a threshold, and saves the binary masks to a new directory.
"""

import numpy as np
import tifffile as tiff
import os
import glob
from pathlib import Path
from natsort import natsorted


def threshold_cellprob_map(cellprob_map, threshold=0.0):
    """
    Threshold a cell probability map.
    
    Parameters:
    -----------
    cellprob_map : numpy.ndarray
        Cell probability map
    threshold : float
        Threshold value (default: 0.0)
        
    Returns:
    --------
    binary_mask : numpy.ndarray
        Binary mask where cellprob > threshold
    """
    binary_mask = (cellprob_map > threshold).astype(np.uint8)
    return binary_mask


def process_directory(input_dir, output_dir, threshold=0.0, pattern="*cellprob*.tif", verbose=True):
    """
    Process all cellprob files in a directory.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing cellprob files
    output_dir : str
        Directory to save thresholded outputs
    threshold : float
        Threshold value (default: 0.0)
    pattern : str
        Glob pattern to match cellprob files (default: *cellprob*.tif)
    verbose : bool
        Enable verbose output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all cellprob files
    cellprob_files = glob.glob(os.path.join(input_dir, pattern))
    cellprob_files = natsorted(cellprob_files)
    
    if len(cellprob_files) == 0:
        print(f"No files found matching pattern '{pattern}' in {input_dir}")
        return
    
    print(f"Found {len(cellprob_files)} cellprob file(s) to threshold")
    print(f"Threshold: {threshold}")
    print(f"Output directory: {output_dir}")
    
    # Process each file
    for i, cellprob_file in enumerate(cellprob_files):
        if verbose and (i % 10 == 0 or i == len(cellprob_files) - 1):
            print(f"Processing {i+1}/{len(cellprob_files)}: {os.path.basename(cellprob_file)}")
        
        # Load cellprob map
        cellprob_map = tiff.imread(cellprob_file)
        
        # Threshold
        binary_mask = threshold_cellprob_map(cellprob_map, threshold)
        
        # Generate output filename
        basename = os.path.basename(cellprob_file)
        output_filename = basename.replace('.tif', f'_thresh{threshold:.3f}.tif')
        if not output_filename.endswith('.tif'):
            output_filename = basename.replace('.tiff', f'_thresh{threshold:.3f}.tiff')
        output_path = os.path.join(output_dir, output_filename)
        
        # Save
        tiff.imwrite(output_path, binary_mask)
        
        if verbose and i == 0:
            print(f"  Input shape: {cellprob_map.shape}, dtype: {cellprob_map.dtype}")
            print(f"  Input range: [{cellprob_map.min():.3f}, {cellprob_map.max():.3f}]")
            print(f"  Output shape: {binary_mask.shape}, dtype: {binary_mask.dtype}")
            print(f"  Pixels above threshold: {binary_mask.sum()} / {binary_mask.size} ({100*binary_mask.sum()/binary_mask.size:.2f}%)")
    
    print(f"\nProcessing complete! Thresholded {len(cellprob_files)} file(s)")


def process_single_file(input_file, output_dir, threshold=0.0, verbose=True):
    """
    Process a single cellprob file.
    
    Parameters:
    -----------
    input_file : str
        Path to cellprob file
    output_dir : str
        Directory to save thresholded output
    threshold : float
        Threshold value (default: 0.0)
    verbose : bool
        Enable verbose output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"Processing file: {input_file}")
        print(f"Threshold: {threshold}")
    
    # Load cellprob map
    cellprob_map = tiff.imread(input_file)
    
    # Threshold
    binary_mask = threshold_cellprob_map(cellprob_map, threshold)
    
    # Generate output filename
    basename = os.path.basename(input_file)
    output_filename = basename.replace('.tif', f'_cellprob_thresh{threshold:.3f}.tif')
    if not output_filename.endswith('.tif'):
        output_filename = basename.replace('.tiff', f'_thresh{threshold:.3f}.tiff')
    output_path = os.path.join(output_dir, output_filename)
    
    # Save
    tiff.imwrite(output_path, binary_mask)
    
    if verbose:
        print(f"Input shape: {cellprob_map.shape}, dtype: {cellprob_map.dtype}")
        print(f"Input range: [{cellprob_map.min():.3f}, {cellprob_map.max():.3f}]")
        print(f"Output shape: {binary_mask.shape}, dtype: {binary_mask.dtype}")
        print(f"Pixels above threshold: {binary_mask.sum()} / {binary_mask.size} ({100*binary_mask.sum()/binary_mask.size:.2f}%)")
        print(f"Saved to: {output_path}")


def main():
    """CLI for thresholding cell probability maps."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Threshold cell probability maps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Threshold all cellprob files in a directory
  python threshold_cellprob.py --input_dir tiles/ --output_dir tiles_thresh/ --threshold 0.5
  
  # Process a single file
  python threshold_cellprob.py --input_file cellprob.tif --output_dir output/ --threshold 2.0
  
  # Use custom pattern
  python threshold_cellprob.py --input_dir tiles/ --output_dir tiles_thresh/ --threshold 0.0 --pattern "*cellprob_blur*.tif"
        '''
    )
    
    parser.add_argument('--input_dir', type=str,
                       help='Directory containing cellprob files')
    parser.add_argument('--input_file', type=str,
                       help='Single cellprob file to process (alternative to --input_dir)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save thresholded outputs')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='Threshold value (default: 0.0)')
    parser.add_argument('--pattern', type=str, default="*cellprob*.tif",
                       help='Glob pattern to match cellprob files (default: *cellprob*.tif)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Check that either input_dir or input_file is provided
    if args.input_dir is None and args.input_file is None:
        parser.error("Either --input_dir or --input_file must be provided")
    
    if args.input_file is not None:
        # Process single file
        process_single_file(
            input_file=args.input_file,
            output_dir=args.output_dir,
            threshold=args.threshold,
            verbose=args.verbose
        )
    else:
        # Process directory
        process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            pattern=args.pattern,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()
