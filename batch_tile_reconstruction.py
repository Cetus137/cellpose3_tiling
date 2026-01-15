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
    tile_id = 1
    for tile_info in tiles:

        print('processing tile number', tile_id)
        tile_id += 1


        z_start = tile_info['z_start']
        z_end = tile_info['z_end']
        y_start = tile_info['y_start']
        y_end = tile_info['y_end']
        x_start = tile_info['x_start']
        x_end = tile_info['x_end']
        
        # Get actual data size (not padded)
        actual_z, actual_y, actual_x = tile_info['original_shape']
        print('original shape:', tile_info['original_shape'])

        tile_dP = tile_info['dP_blur'][:, :actual_z, :actual_y, :actual_x]
        tile_cell_prob = tile_info['cell_prob_blur'][:actual_z, :actual_y, :actual_x]
        
        # Create weight map for this tile (1.0 in center, tapering at edges in overlap regions)
        print('creating weight map...')
        weight_map = create_weight_map(
            (actual_z, actual_y, actual_x),
            overlap_xy,
            z_start, z_end, y_start, y_end, x_start, x_end,
            image_shape
        )
        print('weight map shape:', weight_map.shape)
        
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
    print('timepoint is ', timepoint)
    # Find the specific timepoint files
    flow_files     = [f for f in tif_files if f"timepoint_{timepoint:04d}" in f and "dP"    in f]
    cellprob_files = [f for f in tif_files if f"timepoint_{timepoint:04d}" in f and "cellprob" in f]
    
    if len(flow_files) == 0:
        raise ValueError(f"No flow files found for timepoint {timepoint} in {tile_dir}")
    if len(flow_files) != len(cellprob_files):
        raise ValueError(f"Mismatch: {len(flow_files)} flow files but {len(cellprob_files)} cellprob files")
    
    tiles = []
    max_z, max_y, max_x = 0, 0, 0
    
    # Build tile info and determine image shape
    import re
    for flow_file, cellprob_file in zip(flow_files, cellprob_files):
        # Extract tile metadata from filename using regex
        # Example: ..._z0-256_y0-256_x224-480_...
        m_z = re.search(r'_z(\d+)-(\d+)', flow_file)
        m_y = re.search(r'_y(\d+)-(\d+)', flow_file)
        m_x = re.search(r'_x(\d+)-(\d+)', flow_file)
        if not (m_z and m_y and m_x):
            print(f"Warning: Could not parse tile coordinates from {flow_file}, skipping.")
            continue
        z_start, z_end = int(m_z.group(1)), int(m_z.group(2))
        y_start, y_end = int(m_y.group(1)), int(m_y.group(2))
        x_start, x_end = int(m_x.group(1)), int(m_x.group(2))

        print(f"Found tile: z({z_start}-{z_end}), y({y_start}-{y_end}), x({x_start}-{x_end})")

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
                                 min_size=5000, diameter=None, cellprob_threshold=0.0, do_3D=True, verbose=True):
    dP, cellprob = timepoint_reconstruct_dP_cellprob(
        tile_dir, timepoint, overlap_xy
    )
    if verbose:
        print(f"Reconstructed dP shape: {dP.shape}, cellprob shape: {cellprob.shape}")
    
    # Compute masks from reconstructed dP and cellprob

    if verbose:
        print("Computing masks from reconstructed dP and cellprob...")
    masks = compute_masks(dP,
                         cellprob,
                         min_size=min_size,
                         do_3D=do_3D,
                         cellprob_threshold=cellprob_threshold)
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
    import distutils.util

    parser = argparse.ArgumentParser(description='Reconstruct images from tiled flow and cellprob outputs')
    parser.add_argument('--tile_dir', type=str, required=True,
                        help='Directory containing the tile files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save reconstructed outputs')
    parser.add_argument('--timepoint', type=int, required=True,
                        help='Timepoint number to reconstruct')
    parser.add_argument('--overlap', type=int, default=32,
                        help='Overlap in pixels used during tiling (default: 32)')
    parser.add_argument('--min_size', type=int, default=5000,
                        help='Minimum size of objects to keep (default: 5000)')
    parser.add_argument('--diameter', type=str, default="None",
                        help='Cell diameter for Cellpose model (default: None)')
    parser.add_argument('--cellprob_threshold', type=float, default=0.0,
                        help='Cell probability threshold for segmentation (default: 0.0)')
    parser.add_argument('--do_3D', type=str, default="True",
                        help='Whether to perform 3D segmentation (True/False)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert diameter
    diameter = None if args.diameter == "None" else float(args.diameter)
    # Convert do_3D
    import distutils.util
    do_3D = bool(distutils.util.strtobool(args.do_3D))

    reconstruct_masks_from_tiles(
        tile_dir=args.tile_dir,
        output_dir=args.output_dir,
        timepoint=args.timepoint,
        overlap_xy=args.overlap,
        min_size=args.min_size,
        diameter=diameter,
        cellprob_threshold=args.cellprob_threshold,
        do_3D=do_3D,
        verbose=args.verbose
    )
    print(f"Reconstruction complete.")


if __name__ == "__main__":
    main()
