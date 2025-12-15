from cellpose.models import CellposeModel
import tifffile as tiff
import numpy as np
import scipy.ndimage as ndi
from cellpose.dynamics import compute_masks
from segmentation_3views import segment_zstack_3views
import cellpose
from importlib.metadata import version as _getv
import pkg_resources


def apply_gamma_transform(image, gamma=1.0):
    """
    Apply gamma transformation to an image.
    
    The image is normalized to [0, 1], gamma is applied, then scaled back
    to original range.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (any shape)
    gamma : float
        Gamma value. gamma < 1 brightens, gamma > 1 darkens. Default is 1.0 (no change)
        
    Returns:
    --------
    transformed_image : numpy.ndarray
        Gamma-transformed image in original range
    """
    if gamma == 1.0:
        return image
    
    # Store original range
    original_min = np.min(image)
    original_max = np.max(image)
    original_dtype = image.dtype
    
    # Normalize to [0, 1]
    image_normalized = (image - original_min) / (original_max - original_min)
    
    # Apply gamma transformation
    image_gamma = np.power(image_normalized, gamma)
    
    # Scale back to original range
    image_transformed = image_gamma * (original_max - original_min) + original_min
    
    return image_transformed.astype(original_dtype)


def tile_image_3d_3views(image, tile_size=(256, 256, 256), overlap_xy=32):
    """
    Tile a 3D image into overlapping tiles.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input 3D image with shape (z, y, x)
    tile_size : tuple
        Size of each tile (z, y, x). Default is (256, 256, 256)
    overlap_xy : int
        Overlap in pixels for XY dimensions. Default is 32
        
    Returns:
    --------
    tiles : list of dict
        List of dictionaries containing:
            - 'data': the tile data
            - 'z_start', 'z_end': z coordinates
            - 'y_start', 'y_end': y coordinates  
            - 'x_start', 'x_end': x coordinates
    """
    views, z_size, y_size, x_size = image.shape
    tile_z, tile_y, tile_x = tile_size
    
    tiles = []
    
    # Calculate step sizes (tile size minus overlap)
    step_y = tile_y - overlap_xy
    step_x = tile_x - overlap_xy
    step_z = tile_z  # No overlap in Z yet
    
    # Iterate through Z dimension (no overlap yet)
    for z_start in range(0, z_size, step_z):
        z_end = min(z_start + tile_z, z_size)
        
        # Iterate through Y dimension with overlap
        for y_start in range(0, y_size, step_y):
            y_end = min(y_start + tile_y, y_size)
            
            # Iterate through X dimension with overlap
            for x_start in range(0, x_size, step_x):
                x_end = min(x_start + tile_x, x_size)
                
                # Extract tile
                tile_data = image[:, z_start:z_end, y_start:y_end, x_start:x_end]
                
                # Pad tile if it's smaller than tile_size
                if tile_data.shape[1:] != tile_size:
                    padded_tile = np.zeros((tile_data.shape[0],) + tile_size, dtype=image.dtype)
                    padded_tile[:, :tile_data.shape[1], :tile_data.shape[2], :tile_data.shape[3]] = tile_data
                    tile_data = padded_tile
                
                tiles.append({
                    'data': tile_data,
                    'z_start': z_start,
                    'z_end': z_end,
                    'y_start': y_start,
                    'y_end': y_end,
                    'x_start': x_start,
                    'x_end': x_end,
                    'original_shape': (z_end - z_start, y_end - y_start, x_end - x_start)
                })
    
    return tiles


def reconstruct_from_tiles_3views(tiles, image_shape, overlap_xy=32):
    """
    Reconstruct dP_blur and cell_prob_blur from segmented tiles by averaging overlaps.
    
    Parameters:
    -----------
    tiles : list of dict
        List of tile dictionaries containing segmentation results
    image_shape : tuple
        Shape of the original image (z, y, x)
    overlap_xy : int
        Overlap in pixels for XY dimensions
        
    Returns:
    --------
    dP_blur : numpy.ndarray
        Reconstructed flow field with shape (3, z, y, x)
    cell_prob_blur : numpy.ndarray
        Reconstructed cell probability with shape (z, y, x)
    """
    z_size, y_size, x_size = image_shape
    
    # Initialize output arrays and weight arrays for averaging
    dP_blur = np.zeros((3, z_size, y_size, x_size), dtype=np.float32)
    cell_prob_blur = np.zeros((z_size, y_size, x_size), dtype=np.float32)
    
    dP_weights = np.zeros((3, z_size, y_size, x_size), dtype=np.float32)
    cell_prob_weights = np.zeros((z_size, y_size, x_size), dtype=np.float32)
    
    for tile_info in tiles:
        z_start = tile_info['z_start']
        z_end = tile_info['z_end']
        y_start = tile_info['y_start']
        y_end = tile_info['y_end']
        x_start = tile_info['x_start']
        x_end = tile_info['x_end']
        
        # Get actual data size (not padded)
        actual_z, actual_y, actual_x = tile_info['original_shape']
        
        tile_dP = tile_info['dP_blur'][:, :actual_z, :actual_y, :actual_x]
        tile_cell_prob = tile_info['cell_prob_blur'][:actual_z, :actual_y, :actual_x]
        
        # Create weight map for this tile (1.0 in center, tapering at edges in overlap regions)
        weight_map = create_weight_map(
            (actual_z, actual_y, actual_x),
            overlap_xy,
            z_start, z_end, y_start, y_end, x_start, x_end,
            image_shape
        )
        
        # Accumulate weighted values
        dP_blur[:, z_start:z_end, y_start:y_end, x_start:x_end] += tile_dP * weight_map
        cell_prob_blur[z_start:z_end, y_start:y_end, x_start:x_end] += tile_cell_prob * weight_map
        
        # Accumulate weights
        dP_weights[:, z_start:z_end, y_start:y_end, x_start:x_end] += weight_map
        cell_prob_weights[z_start:z_end, y_start:y_end, x_start:x_end] += weight_map
    
    # Normalize by weights (avoid division by zero)
    dP_weights = np.maximum(dP_weights, 1e-8)
    cell_prob_weights = np.maximum(cell_prob_weights, 1e-8)
    
    dP_blur = dP_blur / dP_weights
    cell_prob_blur = cell_prob_blur / cell_prob_weights
    
    return dP_blur, cell_prob_blur

def create_weight_map(tile_shape, overlap_xy, z_start, z_end, y_start, y_end, x_start, x_end, image_shape):
    """
    Create a weight map for blending tiles in overlap regions.
    Uses linear tapering in overlap regions.
    
    Parameters:
    -----------
    tile_shape : tuple
        Actual shape of the tile data (z, y, x)
    overlap_xy : int
        Overlap in pixels for XY dimensions
    z_start, z_end, y_start, y_end, x_start, x_end : int
        Coordinates of the tile in the full image
    image_shape : tuple
        Shape of the full image (z, y, x)
        
    Returns:
    --------
    weight_map : numpy.ndarray
        Weight map with shape matching tile_shape
    """
    z_tile, y_tile, x_tile = tile_shape
    img_z, img_y, img_x = image_shape
    
    weight_map = np.ones(tile_shape, dtype=np.float32)
    
    # Create linear ramps for overlap regions
    half_overlap = overlap_xy // 2
    
    # Y dimension weights
    # Left edge (if not at image edge)
    if y_start > 0:
        ramp = np.linspace(0, 1, overlap_xy)
        for i in range(min(overlap_xy, y_tile)):
            weight_map[:, i, :] *= ramp[i]
    
    # Right edge (if not at image edge)
    if y_end < img_y:
        ramp = np.linspace(1, 0, overlap_xy)
        for i in range(min(overlap_xy, y_tile)):
            idx = y_tile - overlap_xy + i
            if idx >= 0 and idx < y_tile:
                weight_map[:, idx, :] *= ramp[i]
    
    # X dimension weights
    # Left edge (if not at image edge)
    if x_start > 0:
        ramp = np.linspace(0, 1, overlap_xy)
        for i in range(min(overlap_xy, x_tile)):
            weight_map[:, :, i] *= ramp[i]
    
    # Right edge (if not at image edge)
    if x_end < img_x:
        ramp = np.linspace(1, 0, overlap_xy)
        for i in range(min(overlap_xy, x_tile)):
            idx = x_tile - overlap_xy + i
            if idx >= 0 and idx < x_tile:
                weight_map[:, :, idx] *= ramp[i]
    
    return weight_map


def segment_large_image_3views(image, model, tile_size=(256, 256, 256), overlap_xy=32, 
                       cellpose_config_dict=None, verbose=True):
    """
    Segment a large 3 view 3D image using tiled segmentation.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input 3D image with shape (views, z, y, x)
    model : CellposeModel
        Cellpose model to use for segmentation
    tile_size : tuple
        Size of each tile (views, z, y, x). Default is (256, 256, 256)
    overlap_xy : int
        Overlap in pixels for XY dimensions. Default is 32
    cellpose_config_dict : dict
        Dictionary of cellpose configuration parameters
    verbose : bool
        Print progress information
        
    Returns:
    --------
    dP_blur : numpy.ndarray
        Reconstructed flow field with shape (3, z, y, x)
    cell_prob_blur : numpy.ndarray
        Reconstructed cell probability with shape (z, y, x)
    masks : numpy.ndarray
        Final segmentation masks with shape (z, y, x)
    """
    
    if verbose:
        print(f"Input image shape: {image.shape}")
        print(f"Tile size: {tile_size}")
        print(f"XY overlap: {overlap_xy}")
        print(f'image min: {np.min(image)}, image max: {np.max(image)}')

    
    
    # Check if tiling is necessary
    if all(image.shape[i] <= tile_size[i] for i in range(3)):
        if verbose:
            print("Image fits in single tile, processing without tiling...")
        dP_blur, cell_prob_blur = segment_zstack_3views(image, model, cellpose_config_dict)
    else:
        if verbose:
            print("Image exceeds tile size, proceeding with tiled segmentation...")
        tiles = tile_image_3d_3views(image, tile_size, overlap_xy)
        if verbose:
            print(f"Created {len(tiles)} tiles")
        
        # Segment each tile
        if verbose:
            print("Segmenting tiles...")
        for i, tile_info in enumerate(tiles):
            if verbose:
                print(f"Processing tile {i+1}/{len(tiles)}")
                print(f"  Position: z=[{tile_info['z_start']}:{tile_info['z_end']}], "
                    f"y=[{tile_info['y_start']}:{tile_info['y_end']}], "
                    f"x=[{tile_info['x_start']}:{tile_info['x_end']}]")
            
            # Segment the tile
            dP_blur_tile, cell_prob_blur_tile = segment_zstack_3views(
                tile_info['data'], model, cellpose_config_dict
            )
            
            # Store results in tile_info
            tile_info['dP_blur'] = dP_blur_tile
            tile_info['cell_prob_blur'] = cell_prob_blur_tile
        
        # Reconstruct full image from tiles
        if verbose:
            print("Reconstructing from tiles...")
        dP_blur, cell_prob_blur = reconstruct_from_tiles_3views(tiles, image.shape[1:], overlap_xy)
    
    # Compute final masks on reconstructed data
    if verbose:
        print("Computing final masks...")
    
    default_config = {
        'cell_prob_threshold': 8.0,
    }
    config = {**default_config, **(cellpose_config_dict or {})}
    
    masks  = compute_masks(
        dP_blur, 
        cell_prob_blur, 
        flow_threshold=0.4, 
        min_size=5000, 
        do_3D=True, 
        cellprob_threshold=config['cell_prob_threshold']
    )
    
    if verbose:
        print("Segmentation complete!")
        print(f'masks max: {np.max(masks)}, masks min: {np.min(masks)}')
    
    return dP_blur, cell_prob_blur, masks

def segment_timelapse_3views(video_path, output_dir, model, tile_size=(256, 256, 256), 
                     overlap_xy=32, cellpose_config_dict=None, normalize=True,
                     gamma=1.0, t_list=None, verbose=True, save_timepoint=True):
    """
    Segment a timelapse video (5D: views, time, z, y, x) using tiled segmentation.

    Parameters:
    -----------
    video_path : str
        Path to the input video file
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
    gamma : float
        Gamma value for gamma transformation applied per frame before segmentation.
        gamma < 1 brightens, gamma > 1 darkens. Default is 1.0 (no change)
    t_list : list or None
        Specific timepoint indices to process, e.g., [0, 2, 5, 10].
        If None (default), process all timepoints.
    verbose : bool
        Print progress information
        
    Returns:
    --------
    all_masks : numpy.ndarray
        Segmentation masks for selected timepoints with shape (t, z, y, x)
    """
    import os
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load video
    if verbose:
        print(f"Loading video from {video_path}")
    video = tiff.imread(video_path)
    
    if verbose:
        print(f"Video shape: {video.shape}")
        print(f"number of video.ndim: {video.ndim}")
    
    if video.ndim == 5 :
        print(f"found 5D video (views, time, z, y, x) with shape {video.shape}")
        
    elif video.shape[0] == 3 and video.ndim ==4:
        print("Warning: video has 4 dimensions with 3 views, assuming single timepoint.")
        # Add a time dimension
        video = video[:, np.newaxis, ...]
    else:
        raise ValueError(f"Expected 5D video (views, time, z, y, x) or single timepoint, got shape {video.shape}")
    
    if video.shape[0] != 3:
        raise ValueError(f"Expected 3 views in first dimension, got {video.shape[0]}")

    
    
    n_timepoints = video.shape[1]
    
    # Determine which timepoints to process
    if t_list is None:
        # Process all timepoints
        timepoints_to_process = list(range(n_timepoints))
    else:
        # Process specific timepoints from list
        timepoints_to_process = list(t_list)
        # Validate indices
        if any(t < 0 or t >= n_timepoints for t in timepoints_to_process):
            raise ValueError(f"t_list contains invalid timepoint indices. Valid range: 0-{n_timepoints-1}")
    
    if verbose:
        print(f"Processing {len(timepoints_to_process)} timepoint(s): {timepoints_to_process}")
    
    # Normalize if requested
    if normalize:
        if verbose:
            print("Normalizing video to range 0-1...")
        video = video.astype(np.float32)
        video = (video - np.min(video)) / (np.max(video) - np.min(video))
    
    # Apply gamma transformation if requested
    if gamma is not None:
        if verbose:
            print(f"Applying gamma transformation (gamma={gamma})...")
        # Apply gamma frame by frame (only to timepoints we're processing)
        for view in range(video.shape[0]):
            for t in timepoints_to_process:
                video[view, t] = apply_gamma_transform(video[view, t], gamma=gamma)
    
    # Create filename suffix with gamma parameter
    gamma_suffix = f"_gamma{gamma:.2f}" if gamma is not None else ""

    diam_suffix  = f"_diam{cellpose_config_dict['diameter']:.1f}" if cellpose_config_dict and 'diameter' in cellpose_config_dict and cellpose_config_dict['diameter'] is not None else ""
    
    # Initialize output array
    all_masks = []
    
    # Process each timepoint
    for idx, t in enumerate(timepoints_to_process):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing timepoint {t} ({idx+1}/{len(timepoints_to_process)})")
            print(f"{'='*60}")
        
        # Extract current timepoint
        frame_3views = video[:, t , ...]
        print(f"Frame 3views shape: {frame_3views.shape}")
        
        # Segment the frame
        dP_blur, cell_prob_blur, masks = segment_large_image_3views(
            frame_3views,
            model,
            tile_size=tile_size,
            overlap_xy=overlap_xy,
            cellpose_config_dict=cellpose_config_dict,
            verbose=verbose
        )
        
        # Store results
        all_masks.append(masks)
        
        if save_timepoint:
            # Save individual timepoint mask
            timepoint_prefix = output_dir / f"T{t:04d}{gamma_suffix}{diam_suffix}"
            if verbose:
                print(f"Saving timepoint {t} mask...")
            
            tiff.imwrite(str(timepoint_prefix) + "_masks.tif", masks.astype(np.uint16))
    
    # Convert to numpy array
    all_masks = np.array(all_masks)
    
    return all_masks

def batch_tif_segment_timelapse_3views(input_dir, output_dir, model, file_index=None,
                                   tile_size=(256, 256, 256), overlap_xy=32, 
                                   cellpose_config_dict=None, normalize=True,
                                   gamma=1.0, t_list=None, verbose=True, phrase='restored_timepoint'):
    """
    Batch process all .tif files in a directory for timelapse segmentation.
    
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
    gamma : float
        Gamma value for gamma transformation applied per frame before segmentation.
        gamma < 1 brightens, gamma > 1 darkens. Default is 1.0 (no change)
    t_range : tuple, list, or None
        Time range to process. Can be:
        - None: process all timepoints (default)
    """
    import os
    from pathlib import Path
    from natsort import natsorted
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tif_files = natsorted([f for f in os.listdir(input_dir) if f.lower().endswith('.tif')])
    if verbose:
        print(f"Found {len(tif_files)} .tif files in {input_dir}")

    if phrase is not None:
        tif_files = [f for f in tif_files if phrase in f]
        if verbose:
            print(f"Filtered files with phrase '{phrase}': {len(tif_files)} files remain")

    if file_index is not None:
        if file_index < 0 or file_index >= len(tif_files):
            raise ValueError(f"file_index {file_index} is out of range. Found {len(tif_files)} .tif files.")
        tif_files = [tif_files[file_index]]
        if verbose:
            print(f"Processing only file at index {file_index}: {tif_files[0]}")

    for tif_file in tif_files:
        input_path  = input_dir  / tif_file
        output_path = output_dir / tif_file.replace('.tif', '_segmented.tif')
        if verbose:
            print(f"\nProcessing file: {tif_file}")

        all_masks = segment_timelapse_3views(
            video_path=str(input_path),
            output_dir=str(output_dir),
            model=model,
            tile_size=tile_size,
            overlap_xy=overlap_xy,
            cellpose_config_dict=cellpose_config_dict,
            normalize=normalize,
            gamma=gamma,
            t_list=t_list,
            verbose=verbose,
            save_timepoint=False
        )

        if verbose:
            print(f"Saving all masks to {output_path}")
        tiff.imwrite(str(output_path), all_masks.astype(np.uint16))

    return 

if __name__ == "__main__":
    import argparse
    
    try:
        version = getattr(cellpose, '__version__', None)
        if not version:
            try:
                version = _getv('cellpose')
            except Exception:
                try:
                    version = pkg_resources.get_distribution('cellpose').version
                except Exception:
                    version = 'unknown'
        print('Cellpose version:', version)
    except Exception as e:
        print('Could not determine Cellpose version:', e)

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
    parser.add_argument('--tile_size', nargs=3, type=int, default=[256, 256, 256],
                       help='Tile size (z y x). Default: 256 256 256')
    parser.add_argument('--overlap', type=int, default=32,
                       help='XY overlap in pixels. Default: 32')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Gamma correction value (< 1 brightens, > 1 darkens). Default: 1.0')
    parser.add_argument('--t_list', nargs='+', type=int, metavar='T', default=None,
                       help='Specific timepoints to process. E.g., --t_list 0 5 10 15. If not provided, all timepoints are processed.')
    parser.add_argument('--no_normalize', action='store_true',
                       help='Skip normalization to [0, 1] range')
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Use GPU (default: True)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--diameter', type=float, default=None,
                       help='Cell diameter for Cellpose model (default: None)')
    
    args = parser.parse_args()
    
    # Check if running with command-line arguments or using example code
    if args.input_dir and args.output_dir and args.model:
        # Command-line mode
        print("Running in command-line mode...")
        
        # Print timepoint info
        if args.t_list is not None:
            print(f"Processing specific timepoints: {args.t_list}")
        else:
            print("Processing all timepoints")
        
        # Load model
        print(f"Loading model from {args.model}")
        model = CellposeModel(gpu=args.gpu, pretrained_model=args.model)
        
        cellpose_config = {
            'diameter': args.diameter,
        }

        # Run segmentation
        all_masks = batch_tif_segment_timelapse_3views(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            file_index=args.file_index,
            model=model,
            tile_size=tuple(args.tile_size),
            overlap_xy=args.overlap,
            gamma=args.gamma,
            t_list=args.t_list,
            cellpose_config_dict=cellpose_config,
            normalize=not args.no_normalize,
            verbose=args.verbose
        )
        
        print(f"\nSegmentation complete!")


    