from cellpose.models import CellposeModel
import tifffile as tiff
import numpy as np
import scipy.ndimage as ndi
from cellpose.dynamics import compute_masks
from segmentation import segment_zstack
import cellpose
from importlib.metadata import version as _getv
import pkg_resources


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
    
    img_path = r"/Users/ewheeler/cellpose3_testing/data/T0_32bit_xy.tif"
    output_path = r"/Users/ewheeler/cellpose3_testing/data/T0_32bit_xy_tiled_segmented.tif"
    
    vid = tiff.imread(img_path)
    print('Video shape:', vid.shape)
    
    # Normalize the video to range 0-1
    vid = vid.astype(np.float32)
    vid = (vid - np.min(vid)) / (np.max(vid) - np.min(vid))
    
    # Segment with tiling
    # Adjust tile_size and overlap_xy as needed
    dP_blur, cell_prob_blur, masks = segment_large_image(
        vid, 
        model,
        tile_size=(256, 128, 128),
        overlap_xy=32,  # Tunable overlap parameter
        verbose=True
    )
    
    # Save results
    print(f"Saving masks to {output_path}")
    tiff.imwrite(output_path, masks.astype(np.uint16))
