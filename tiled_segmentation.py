from cellpose.models import CellposeModel
import tifffile as tiff
import numpy as np
import scipy.ndimage as ndi
from cellpose.dynamics import compute_masks
from segmentation import segment_zstack
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


def tile_image_3d(image, tile_size=(256, 256, 256), overlap_xy=32):
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
    z_size, y_size, x_size = image.shape
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
                tile_data = image[z_start:z_end, y_start:y_end, x_start:x_end]
                
                # Pad tile if it's smaller than tile_size
                if tile_data.shape != tile_size:
                    padded_tile = np.zeros(tile_size, dtype=image.dtype)
                    padded_tile[:tile_data.shape[0], :tile_data.shape[1], :tile_data.shape[2]] = tile_data
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


def reconstruct_from_tiles(tiles, image_shape, overlap_xy=32):
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


def segment_large_image(image, model, tile_size=(256, 256, 256), overlap_xy=32, 
                       cellpose_config_dict=None, verbose=True):
    """
    Segment a large 3D image using tiled segmentation.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input 3D image with shape (z, y, x)
    model : CellposeModel
        Cellpose model to use for segmentation
    tile_size : tuple
        Size of each tile (z, y, x). Default is (256, 256, 256)
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
    
    # Check if tiling is necessary
    if all(image.shape[i] <= tile_size[i] for i in range(3)):
        if verbose:
            print("Image fits in single tile, processing without tiling...")
        return segment_zstack(image, model, cellpose_config_dict)
    
    # Tile the image
    if verbose:
        print("Tiling image...")
    tiles = tile_image_3d(image, tile_size, overlap_xy)
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
        dP_blur_tile, cell_prob_blur_tile = segment_zstack(
            tile_info['data'], model, cellpose_config_dict
        )
        
        # Store results in tile_info
        tile_info['dP_blur'] = dP_blur_tile
        tile_info['cell_prob_blur'] = cell_prob_blur_tile
    
    # Reconstruct full image from tiles
    if verbose:
        print("Reconstructing from tiles...")
    dP_blur, cell_prob_blur = reconstruct_from_tiles(tiles, image.shape, overlap_xy)
    
    # Compute final masks on reconstructed data
    if verbose:
        print("Computing final masks...")
    
    default_config = {
        'cell_prob_threshold': 0.0,
    }
    config = {**default_config, **(cellpose_config_dict or {})}
    
    masks, p = compute_masks(
        dP_blur, 
        cell_prob_blur, 
        flow_threshold=0.4, 
        min_size=5000, 
        do_3D=True, 
        cellprob_threshold=config['cell_prob_threshold']
    )
    
    if verbose:
        print("Segmentation complete!")
    
    return dP_blur, cell_prob_blur, masks


def segment_timelapse(video_path, output_dir, model, tile_size=(256, 256, 256), 
                     overlap_xy=32, cellpose_config_dict=None, normalize=True,
                     gamma=1.0, verbose=True):
    """
    Segment a timelapse video (4D: time, z, y, x) using tiled segmentation.
    
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
    verbose : bool
        Print progress information
        
    Returns:
    --------
    all_masks : numpy.ndarray
        Segmentation masks for all timepoints with shape (t, z, y, x)
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
    
    # Determine if video is 4D (timelapse) or 3D (single timepoint)
    if video.ndim == 3:
        # Single timepoint, add time dimension
        video = video[np.newaxis, ...]
        is_timelapse = False
        if verbose:
            print("Single timepoint detected, processing as single frame")
    elif video.ndim == 4:
        is_timelapse = True
        if verbose:
            print(f"Timelapse detected with {video.shape[0]} timepoints")
    else:
        raise ValueError(f"Expected 3D or 4D video, got shape {video.shape}")
    
    n_timepoints = video.shape[0]
    
    # Normalize if requested
    if normalize:
        if verbose:
            print("Normalizing video to range 0-1...")
        video = video.astype(np.float32)
        video = (video - np.min(video)) / (np.max(video) - np.min(video))
    
    # Apply gamma transformation if requested
    if gamma != 1.0:
        if verbose:
            print(f"Applying gamma transformation (gamma={gamma})...")
        # Apply gamma frame by frame
        for t in range(n_timepoints):
            video[t] = apply_gamma_transform(video[t], gamma=gamma)
    
    # Create filename suffix with gamma parameter
    gamma_suffix = f"_gamma{gamma:.2f}" if gamma != 1.0 else ""
    
    # Initialize output array
    all_masks = []
    
    # Process each timepoint
    for t in range(n_timepoints):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing timepoint {t+1}/{n_timepoints}")
            print(f"{'='*60}")
        
        # Extract current timepoint
        frame = video[t]
        
        # Segment the frame
        dP_blur, cell_prob_blur, masks = segment_large_image(
            frame,
            model,
            tile_size=tile_size,
            overlap_xy=overlap_xy,
            cellpose_config_dict=cellpose_config_dict,
            verbose=verbose
        )
        
        # Store results
        all_masks.append(masks)
        
        # Save individual timepoint mask
        timepoint_prefix = output_dir / f"T{t:04d}{gamma_suffix}"
        if verbose:
            print(f"Saving timepoint {t} mask...")
        
        tiff.imwrite(str(timepoint_prefix) + "_masks.tif", masks.astype(np.uint16))
    
    # Convert to numpy array
    all_masks = np.array(all_masks)
    
    # Save complete video results
    if verbose:
        print(f"\n{'='*60}")
        print("Saving complete video results...")
        print(f"{'='*60}")
    
    video_output_prefix = output_dir / f"video_complete{gamma_suffix}"
    tiff.imwrite(str(video_output_prefix) + "_masks.tif", all_masks.astype(np.uint16))
    
    if verbose:
        print(f"\nAll results saved to {output_dir}")
        print(f"Total timepoints processed: {n_timepoints}")
    
    return all_masks


if __name__ == "__main__":
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

    # Example usage
    pretrained_model_path = r'/Users/ewheeler/.cellpose/models/CP_20250430_181517'
    model = CellposeModel(gpu=True, pretrained_model=pretrained_model_path)
    
    

    # Example: Process timelapse video with gamma transformation
    video_path = r"/Users/ewheeler/cellpose3_testing/data/timelapse_video.tif"
    output_dir = r"/Users/ewheeler/cellpose3_testing/data/timelapse_results"
    
    all_masks = segment_timelapse(
        video_path=video_path,
        output_dir=output_dir,
        model=model,
        tile_size=(256, 128, 128),
        overlap_xy=32,
        gamma=0.8,  # Apply gamma correction (< 1 brightens, > 1 darkens, 1.0 = no change)
        verbose=True
    )
    
    print(f"\nFinal output shape: {all_masks.shape}")
    