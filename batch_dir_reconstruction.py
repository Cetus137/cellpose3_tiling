import numpy as np
from cellpose.dynamics import compute_masks
import tifffile as tiff
import os
from natsort import natsorted
from pathlib import Path


def find_file_pairs(input_dir, verbose=True):
    """
    Find pairs of files ending in cellprob_blur and dP_blur.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the files
    verbose : bool
        Print verbose output
        
    Returns:
    --------
    pairs : list of tuples
        List of (cellprob_file, dP_file) tuples
    """
    files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    files = natsorted(files)
    
    # Find cellprob files
    cellprob_files = [f for f in files if 'cellprob_blur' in f]
    
    pairs = []
    for cellprob_file in cellprob_files:
        # Construct corresponding dP file name
        # Replace cellprob_blur with dP_blur
        dP_file = cellprob_file.replace('cellprob_blur', 'dP_blur')
        
        if dP_file in files:
            pairs.append((cellprob_file, dP_file))
            #if verbose:
            #    print(f"Found pair: {cellprob_file} <-> {dP_file}")
        else:
            if verbose:
                print(f"Warning: No matching dP file for {cellprob_file}")
    
    return pairs


def process_file_pair(cellprob_path, dP_path, output_dir, cellpose_config, verbose=True):
    """
    Process a single pair of cellprob_blur and dP_blur files.
    
    Parameters:
    -----------
    cellprob_path : str
        Path to cellprob_blur file
    dP_path : str
        Path to dP_blur file
    output_dir : str
        Directory to save output masks
    cellpose_config : dict
        Dictionary containing cellpose configuration:
            - min_size: minimum object size (default: 5000)
            - do_3D: whether to do 3D segmentation (default: True)
            - cellprob_threshold: cell probability threshold (default: 0.0)
    verbose : bool
        Print verbose output
        
    Returns:
    --------
    output_path : str
        Path to saved mask file
    """
    if verbose:
        print(f"\nProcessing pair:")
        print(f"  cellprob: {cellprob_path}")
        print(f"  dP: {dP_path}")
    
    # Load data
    cellprob = tiff.imread(cellprob_path)
    dP = tiff.imread(dP_path)
    
    if verbose:
        print(f"  cellprob shape: {cellprob.shape}")
        print(f"  dP shape: {dP.shape}")
    
    # Extract config parameters
    min_size = cellpose_config.get('min_size', 5000)
    do_3D = cellpose_config.get('do_3D', True)
    cellprob_threshold = cellpose_config.get('cellprob_threshold', 0.0)
    
    # Compute masks
    if verbose:
        print(f"  Computing masks (min_size={min_size}, do_3D={do_3D}, threshold={cellprob_threshold})...")
    
    masks = compute_masks(
        dP,
        cellprob,
        min_size=min_size,
        do_3D=do_3D,
        cellprob_threshold=cellprob_threshold
    )
    
    if verbose:
        print(f"  Masks shape: {masks.shape}")
        print(f"  Unique labels: {len(np.unique(masks)) - 1}")  # -1 to exclude background
    
    # Generate output filename
    base_name = os.path.basename(cellprob_path)
    # Remove cellprob_blur suffix and add _masks
    output_name = base_name.replace('_cellprob_blur.tif', '.tif')
    output_path = os.path.join(output_dir, output_name)
    
    # Save masks
    if verbose:
        print(f"  Saving to: {output_path}")
    
    tiff.imwrite(output_path, masks.astype(np.uint16))
    
    return output_path


def batch_process_directory(input_dir, output_dir, cellpose_config=None, verbose=True, file_index=None):
    """
    Process all file pairs in a directory.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing cellprob_blur and dP_blur files
    output_dir : str
        Directory to save output masks
    cellpose_config : dict
        Dictionary containing cellpose configuration:
            - min_size: minimum object size (default: 5000)
            - do_3D: whether to do 3D segmentation (default: True)
            - cellprob_threshold: cell probability threshold (default: 0.0)
    verbose : bool
        Print verbose output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Default config
    if cellpose_config is None:
        cellpose_config = {
            'min_size': 5000,
            'do_3D': True,
            'cellprob_threshold': 0.0
        }
    
    if verbose:
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Cellpose config: {cellpose_config}")
    
    # Find file pairs
    pairs = find_file_pairs(input_dir, verbose=verbose)
    
    if len(pairs) == 0:
        print("Warning: No file pairs found!")
        return
    

    
    print(f"\nFound {len(pairs)} file pair(s) to process")
    print(file_index)
    #only process specific file if file_index is given
    if file_index is not None:
        if 0 <= file_index < len(pairs):
            pairs = [pairs[file_index]]
            if verbose:
                print(f"Processing only file pair at index {file_index}")
                print(f"Selected pair: {pairs[0][0]} <-> {pairs[0][1]}")
        else:
            print(f"Error: file_index {file_index} is out of range (0 to {len(pairs)-1})")
            return
    
    # Process each pair
    for i, (cellprob_file, dP_file) in enumerate(pairs):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing pair {i+1}/{len(pairs)}")
            print(f"{'='*60}")
        
        cellprob_path = os.path.join(input_dir, cellprob_file)
        dP_path = os.path.join(input_dir, dP_file)
        
        output_path = process_file_pair(
            cellprob_path,
            dP_path,
            output_dir,
            cellpose_config,
            verbose=verbose
        )
        
        if verbose:
            print(f"Successfully saved: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete! Processed {len(pairs)} file pairs.")
    print(f"{'='*60}")


def main():
    """
    CLI for batch processing directory of cellprob_blur and dP_blur files.
    """
    import argparse
    import distutils.util

    parser = argparse.ArgumentParser(
        description='Batch process cellprob_blur and dP_blur file pairs to generate masks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python batch_dir_reconstruction.py \\
        --input_dir /path/to/tiles \\
        --output_dir /path/to/masks \\
        --min_size 5000 \\
        --cellprob_threshold 0.0 \\
        --do_3D True \\
        --verbose
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing cellprob_blur and dP_blur files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output masks')
    parser.add_argument('--min_size', type=int, default=5000,
                        help='Minimum size of objects to keep (default: 5000)')
    parser.add_argument('--cellprob_threshold', type=float, default=0.0,
                        help='Cell probability threshold for segmentation (default: 0.0)')
    parser.add_argument('--do_3D', type=str, default="True",
                        help='Whether to perform 3D segmentation (True/False, default: True)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--file_index', type=int, default=None,
                        help='Index of specific file pair to process (default: process all)')

    args = parser.parse_args()

    # Convert do_3D
    do_3D = bool(distutils.util.strtobool(args.do_3D))

    # Build config
    cellpose_config = {
        'min_size': args.min_size,
        'do_3D': do_3D,
        'cellprob_threshold': args.cellprob_threshold
    }

    print(f"Starting batch processing with config: {cellpose_config}")
    print(f"Input directory: {args.input_dir}" )
    print(f"Output directory: {args.output_dir}" )
    print(f"File index: {args.file_index}" )

    # Run batch processing
    batch_process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cellpose_config=cellpose_config,
        file_index=args.file_index,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
