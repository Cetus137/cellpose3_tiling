from cellpose.models import CellposeModel
import tifffile as tiff
import numpy as np
import scipy.ndimage as ndi
from cellpose.dynamics import compute_masks
from segmentation_3views import segment_zstack_3views
import cellpose


def batch_tile_segment_3views(input_dir, output_dir,model,file_index=None,
                              cellpose_config_dict=None, normalize=True,verbose=True):
    """
    Batch process all .tif files in a directory for tiled segmentation.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing input .tif files
    output_dir : str
        Directory to save output files
    model : CellposeModel
        Cellpose model to use for segmentation
    tile_size : tuple
        Size of each tile (z, y, x). Default is (256, 256, 256)
    overlap_xy : int
        Overlap in pixels for XY dimensions. Default is 32
    cellpose_config_dict : dict
        Dictionary of cellpose configuration parameters
    normalize : bool
        Whether to normalize the video to range 0-1. Default is True
    """
    import os
    from pathlib import Path
    from natsort import natsorted
       
        
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(output_dir)
    
    # Count all segmentation files in directory and subdirectories using os.walk
    tif_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if (file.endswith('.tif') or file.endswith('.tiff')) and 'tile' in file:
                tif_files.append(os.path.join(root, file))

    if file_index is not None:
        if file_index < 0 or file_index >= len(tif_files):
            raise ValueError(f"file_index {file_index} is out of range. Found {len(tif_files)} .tif files.")
        tif_files = [tif_files[file_index]]
        if verbose:
            print(f"Processing only file at index {file_index}: {tif_files[0]}")

    for tif_file in tif_files:
        input_path  = input_dir  / tif_file
        if verbose:
            print(f"\nProcessing file: {tif_file}")
        
        image = tiff.imread(str(input_path))

        # Normalize if requested
        if normalize:
            if verbose:
                print("Normalizing video to range 0-1...")
            image = image.astype(np.float32)
            image = (image - np.min(image)) / (np.max(image) - np.min(image))

        dP_blur, cell_prob_blur = segment_zstack_3views(image, model, cellpose_config_dict)
    
        tif_file_basename = os.path.basename(tif_file)
        #save the outputs accordingly
        output_dP_path       = output_dir / tif_file_basename.replace('.tif', '_dP_blur.tif')
        output_cellprob_path = output_dir / tif_file_basename.replace('.tif', '_cellprob_blur.tif')

        print(f"Saving dP_blur to {output_dP_path}")
        print(f"Saving cell_prob_blur to {output_cellprob_path}")

        tiff.imwrite(str(output_dP_path), dP_blur.astype(np.float32))
        tiff.imwrite(str(output_cellprob_path), cell_prob_blur.astype(np.float32))

    return 


if __name__ == "__main__":
    import argparse

    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description='Tiled 3D segmentation with Cellpose for large timelapses',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
                Examples:
                # Segment all timepoints with default settings
                python tiled_segmentation.py --video video.tif --output results/ --model /path/to/model
                
                # Segment with gamma correction
                python tiled_segmentation.py --video video.tif --output results/ --model /path/to/model --gamma 0.8
                
                # Segment specific timepoints
                python tiled_segmentation.py --video video.tif --output results/ --model /path/to/model --t_list 0 5 10 15
                
                # Custom tile size and overlap
                python tiled_segmentation.py --video video.tif --output results/ --model /path/to/model --tile_size 128 128 128 --overlap 64
                        '''
                    )
    
    parser.add_argument('--input_dir', type=str, help='Path to input directory containing .tif files')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--file_index', type=int, default=None, help='Index of specific file to process from input directory')
    parser.add_argument('--model', type=str, help='Path to pretrained Cellpose model')
    parser.add_argument('--no_normalize', action='store_true',
                       help='Skip normalization to [0, 1] range')
    parser.add_argument('--gpu', action='store_true', default=False,
                       help='Use GPU (default: True)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--diameter', type=float, default=None,
                       help='Cell diameter for Cellpose model (default: None)')
    parser.add_argument('--cellprob_threshold', type=float, default=0.0,
                       help='Cell probability threshold for Cellpose (default: 0.0)')
    
    args = parser.parse_args()
    
    # Check if running with command-line arguments or using example code
    if args.input_dir and args.output_dir and args.model:
        # Command-line mode
        print("Running in command-line mode...")
        
        
        # Load model
        print(f"Loading model from {args.model}")
        model = CellposeModel(gpu=args.gpu, pretrained_model=args.model)
        
        cellpose_config = {
            'diameter': args.diameter,
            'cell_prob_threshold': args.cellprob_threshold,
            'use_gpu': args.gpu
        }

        # Run segmentation
        batch_tile_segment_3views(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            file_index=args.file_index,
            model=model,
            cellpose_config_dict=cellpose_config,
            normalize=not args.no_normalize,
            verbose=args.verbose
        )
        
        print(f"\nSegmentation complete!")


    