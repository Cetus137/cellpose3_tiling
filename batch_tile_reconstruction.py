import numpy as np
from cellpose.dynamics import compute_masks
import tifffile as tiff
import os

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

def timepoint_reconstruct_dP_cellprob(tile_dir, timepoint, overlap_xy=32):
    """
    Reconstruct dP_blur and cell_prob_blur for a specific timepoint from tiled segmentations.
    
    Parameters:
    -----------
    tile_dir : str
        Directory containing the tile files
    timepoint : int
        Timepoint number to reconstruct
    overlap_xy : int
        Overlap in pixels used during tiling (default: 32)
        
    Returns:
    --------
    dP_blur : numpy.ndarray
        Reconstructed flow field with shape (3, z, y, x)
    cell_prob_blur : numpy.ndarray
        Reconstructed cell probability with shape (z, y, x)
    image_shape : tuple
        Shape of the reconstructed image (z, y, x)
    """
    import os 
    from natsort import natsorted
    import tifffile as tiff

    # Find all tif files in the directory
    tif_files = [f for f in os.listdir(tile_dir) if f.endswith('.tif')]
    tif_files = natsorted(tif_files)
    
    # Find the specific timepoint files
    flow_files     = [f for f in tif_files if f"timepoint_{timepoint:04d}" in f and "flows"    in f]
    cellprob_files = [f for f in tif_files if f"timepoint_{timepoint:04d}" in f and "cellprob" in f]
    
    if len(flow_files) == 0:
        raise ValueError(f"No flow files found for timepoint {timepoint} in {tile_dir}")
    if len(flow_files) != len(cellprob_files):
        raise ValueError(f"Mismatch: {len(flow_files)} flow files but {len(cellprob_files)} cellprob files")
    
    tiles = []
    max_z, max_y, max_x = 0, 0, 0
    
    # Build tile info and determine image shape
    for flow_file, cellprob_file in zip(flow_files, cellprob_files):
        # Extract tile metadata from filename
        parts = flow_file.split('_')
        z_start = int(parts[parts.index('zstart') + 1])
        z_end   = int(parts[parts.index('zend')   + 1])
        y_start = int(parts[parts.index('ystart') + 1])
        y_end   = int(parts[parts.index('yend')   + 1])
        x_start = int(parts[parts.index('xstart') + 1])
        x_end   = int(parts[parts.index('xend')   + 1])
        
        # Track maximum extents to determine image shape
        max_z = max(max_z, z_end)
        max_y = max(max_y, y_end)
        max_x = max(max_x, x_end)
        
        # Load the actual data
        flow_data = tiff.imread(os.path.join(tile_dir, flow_file))
        cellprob_data = tiff.imread(os.path.join(tile_dir, cellprob_file))
        
        tiles.append({
            'dP_blur': flow_data,
            'cell_prob_blur': cellprob_data,
            'z_start': z_start,
            'z_end': z_end,
            'y_start': y_start,
            'y_end': y_end,
            'x_start': x_start,
            'x_end': x_end,
            'original_shape': (z_end - z_start, y_end - y_start, x_end - x_start)
        })
    
    image_shape = (max_z, max_y, max_x)
    print(f"Reconstructing timepoint {timepoint} with {len(tiles)} tiles, image shape: {image_shape}")
    
    # Reconstruct using the existing infrastructure
    dP_blur, cell_prob_blur = reconstruct_from_tiles_3views(tiles, image_shape, overlap_xy)
    
    return dP_blur, cell_prob_blur


def reconstruct_masks_from_tiles(tile_dir, output_dir, timepoint, overlap_xy=32,
                                 cellpose_config_dict=None, verbose=True):

        dP, cellprob = timepoint_reconstruct_dP_cellprob(
            tile_dir, timepoint, overlap_xy
        )


        masks = compute_masks(dP, 
                             cellprob, 
                             flow_threshold=cellpose_config_dict.get('flow_threshold', 0.4), 
                             min_size=cellpose_config_dict.get('min_size', 15), 
                             do_3D=cellpose_config_dict.get('do_3D', True), 
                             cellprob_threshold=cellpose_config_dict.get('cellprob_threshold', 0.0))
        if verbose:
            print(f"Reconstructed masks for timepoint {timepoint} with shape {masks.shape}")
            print(f"Unique labels in masks: {np.unique(masks)}")
            
        # Save masks
        output_path = os.path.join(output_dir, f'restored_timepoint_{timepoint:04d}_segmented.tif')
        print(f"Saving masks to {output_path}")
        tiff.imwrite(output_path, masks.astype(np.uint16))
        return


def main():
    """
    CLI for reconstructing full images from tiles.
    """
    import argparse
    import tifffile as tiff
    import os
    
    parser = argparse.ArgumentParser(description='Reconstruct images from tiled flow and cellprob outputs')
    parser.add_argument('--tile_dir', type=str, required=True,
                        help='Directory containing the tile files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save reconstructed outputs')
    parser.add_argument('--timepoint', type=int, required=True,
                        help='Timepoint number to reconstruct')
    parser.add_argument('--overlap', type=int, default=32,
                        help='Overlap in pixels used during tiling (default: 32)')
    parser.add_argument('--save_flows', action='store_true',
                        help='Save reconstructed flow field')
    parser.add_argument('--save_cellprob', action='store_true',
                        help='Save reconstructed cellprob')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Reconstruct
    dP_blur, cell_prob_blur, image_shape = timepoint_reconstruct_dP_cellprob(
        args.tile_dir, 
        args.timepoint, 
        args.overlap
    )
    
    # Save outputs
    if args.save_flows:
        flow_path = os.path.join(args.output_dir, f"timepoint_{args.timepoint:04d}_flows.tif")
        tiff.imwrite(flow_path, dP_blur.astype(np.float32))
        print(f"Saved flows to {flow_path}")
    
    if args.save_cellprob:
        cellprob_path = os.path.join(args.output_dir, f"timepoint_{args.timepoint:04d}_cellprob.tif")
        tiff.imwrite(cellprob_path, cell_prob_blur.astype(np.float32))
        print(f"Saved cellprob to {cellprob_path}")
    
    print(f"Reconstruction complete. Image shape: {image_shape}")


if __name__ == "__main__":
    main()
