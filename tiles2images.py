#!/usr/bin/env python3
"""
tiles2images.py - Reconstruct full images from tiles created by image2tiles.py

This script reverses the tiling process by reading tiles from a directory,
grouping them by timepoint, extracting their positions from filenames,
and reconstructing the original full images.
"""

import numpy as np
import tifffile as tiff
import os
import re
from pathlib import Path
from collections import defaultdict
from natsort import natsorted
import glob


def parse_tile_filename(filename):
    """
    Parse tile filename to extract position information.
    
    Expected format: {basename}_tile_{i:04d}_z{z_start}-{z_end}_y{y_start}-{y_end}_x{x_start}-{x_end}.tif
    
    Parameters:
    -----------
    filename : str
        Tile filename
        
    Returns:
    --------
    dict : Dictionary containing:
        - basename: base name without tile info
        - tile_idx: tile index
        - z_start, z_end: z coordinates
        - y_start, y_end: y coordinates
        - x_start, x_end: x coordinates
    """
    basename = os.path.basename(filename)
    
    # Pattern to match: {basename}_tile_{idx}_z{start}-{end}_y{start}-{end}_x{start}-{end}[anything].tif[f]
    # The .* matches any characters (or none) between coordinates and extension
    # \.tiff?$ matches .tif or .tiff at the end of the filename
    pattern = r'(.+?)_tile_(\d+)_z(\d+)-(\d+)_y(\d+)-(\d+)_x(\d+)-(\d+).*\.tiff?$'
    match = re.match(pattern, basename)
    
    if not match:
        raise ValueError(f"Filename doesn't match expected tile format: {filename}")
    
    return {
        'basename': match.group(1),
        'tile_idx': int(match.group(2)),
        'z_start': int(match.group(3)),
        'z_end': int(match.group(4)),
        'y_start': int(match.group(5)),
        'y_end': int(match.group(6)),
        'x_start': int(match.group(7)),
        'x_end': int(match.group(8))
    }


def group_tiles_by_timepoint(tile_dir, pattern="*_tile_*.tif", marker=None):
    """
    Group tile files by their timepoint/basename.
    
    Parameters:
    -----------
    tile_dir : str
        Directory containing tile files
    pattern : str
        Glob pattern to match tile files
    marker : str, optional
        If provided, only include files containing this marker in their filename
        (e.g., 'cellprob', 'dP', 'magnitude')
        
    Returns:
    --------
    dict : Dictionary mapping basename to list of tile file paths
    """
    tile_files = glob.glob(os.path.join(tile_dir, pattern))
    
    if len(tile_files) == 0:
        raise ValueError(f"No tile files found in {tile_dir} matching pattern {pattern}")
    
    # Group by basename
    grouped = defaultdict(list)
    for tile_file in tile_files:
        # Filter by marker if provided
        if marker is not None and marker not in os.path.basename(tile_file):
            continue
        
        try:
            info = parse_tile_filename(tile_file)
            grouped[info['basename']].append(tile_file)
        except ValueError as e:
            print(f"Warning: Skipping file {tile_file}: {e}")
            continue
    
    # Sort tiles within each group
    for basename in grouped:
        grouped[basename] = natsorted(grouped[basename])
    
    return dict(grouped)


def reconstruct_from_tiles(tile_files, overlap_xy=32, verbose=True):
    """
    Reconstruct a full image from tiles.
    
    Parameters:
    -----------
    tile_files : list
        List of tile file paths for a single timepoint
    overlap_xy : int
        Overlap in pixels used during tiling (default: 32)
    verbose : bool
        Enable verbose output
        
    Returns:
    --------
    reconstructed : numpy.ndarray
        Reconstructed full image
    """
    if verbose:
        print(f"Reconstructing from {len(tile_files)} tiles")
    
    # Parse all tile files to get positions and determine full image shape
    tile_info_list = []
    max_z, max_y, max_x = 0, 0, 0
    
    for tile_file in tile_files:
        info = parse_tile_filename(tile_file)
        info['filepath'] = tile_file
        tile_info_list.append(info)
        
        max_z = max(max_z, info['z_end'])
        max_y = max(max_y, info['y_end'])
        max_x = max(max_x, info['x_end'])
    
    # Determine image shape
    image_shape = (max_z, max_y, max_x)
    
    if verbose:
        print(f"Reconstructed image shape will be: {image_shape}")
    
    # Load first tile to get dtype and check for additional dimensions
    first_tile = tiff.imread(tile_info_list[0]['filepath'])
    first_tile = np.squeeze(first_tile)
    
    # Determine if tiles have additional dimensions (e.g., views, channels)
    if first_tile.ndim == 4:
        # Has extra dimension (e.g., views)
        n_views = first_tile.shape[0]
        reconstructed = np.zeros((n_views,) + image_shape, dtype=first_tile.dtype)
        weights = np.zeros((n_views,) + image_shape, dtype=np.float32)
        has_extra_dim = True
    else:
        # Standard 3D
        reconstructed = np.zeros(image_shape, dtype=first_tile.dtype)
        weights = np.zeros(image_shape, dtype=np.float32)
        has_extra_dim = False
    
    if verbose:
        print(f"Tile data has extra dimension: {has_extra_dim}")
        if has_extra_dim:
            print(f"Number of views/channels: {n_views}")
    
    # Reconstruct by averaging overlapping regions
    for i, info in enumerate(tile_info_list):
        if verbose and i % 10 == 0:
            print(f"Processing tile {i+1}/{len(tile_info_list)}")
        
        # Load tile
        tile_data = tiff.imread(info['filepath'])
        tile_data = np.squeeze(tile_data)
        
        # Get actual extent (without padding)
        z_start, z_end = info['z_start'], info['z_end']
        y_start, y_end = info['y_start'], info['y_end']
        x_start, x_end = info['x_start'], info['x_end']
        
        actual_z = z_end - z_start
        actual_y = y_end - y_start
        actual_x = x_end - x_start
        
        # Extract only the non-padded portion
        if has_extra_dim:
            tile_data = tile_data[:, :actual_z, :actual_y, :actual_x]
        else:
            tile_data = tile_data[:actual_z, :actual_y, :actual_x]
        
        # Create weight map for blending in overlap regions
        weight_map = create_weight_map(
            (actual_z, actual_y, actual_x),
            overlap_xy,
            z_start, z_end, y_start, y_end, x_start, x_end,
            image_shape
        )
        
        # Accumulate weighted data
        if has_extra_dim:
            reconstructed[:, z_start:z_end, y_start:y_end, x_start:x_end] += tile_data * weight_map
            weights[:, z_start:z_end, y_start:y_end, x_start:x_end] += weight_map
        else:
            reconstructed[z_start:z_end, y_start:y_end, x_start:x_end] += tile_data * weight_map
            weights[z_start:z_end, y_start:y_end, x_start:x_end] += weight_map
    
    # Normalize by weights
    weights = np.maximum(weights, 1e-8)  # Avoid division by zero
    reconstructed = reconstructed / weights
    
    return reconstructed.astype(first_tile.dtype)


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
    z_size, y_size, x_size = tile_shape
    full_z, full_y, full_x = image_shape
    
    # Initialize with ones
    weight_map = np.ones((z_size, y_size, x_size), dtype=np.float32)
    
    # Taper in Y dimension if there's overlap
    if y_start > 0 and overlap_xy > 0:
        # Taper at the beginning
        taper = np.linspace(0, 1, overlap_xy)
        for i in range(min(overlap_xy, y_size)):
            weight_map[:, i, :] *= taper[i]
    
    if y_end < full_y and overlap_xy > 0:
        # Taper at the end
        taper = np.linspace(1, 0, overlap_xy)
        for i in range(min(overlap_xy, y_size)):
            idx = y_size - overlap_xy + i
            if idx >= 0 and idx < y_size:
                weight_map[:, idx, :] *= taper[i]
    
    # Taper in X dimension if there's overlap
    if x_start > 0 and overlap_xy > 0:
        # Taper at the beginning
        taper = np.linspace(0, 1, overlap_xy)
        for i in range(min(overlap_xy, x_size)):
            weight_map[:, :, i] *= taper[i]
    
    if x_end < full_x and overlap_xy > 0:
        # Taper at the end
        taper = np.linspace(1, 0, overlap_xy)
        for i in range(min(overlap_xy, x_size)):
            idx = x_size - overlap_xy + i
            if idx >= 0 and idx < x_size:
                weight_map[:, :, idx] *= taper[i]
    
    return weight_map


def reconstruct_directory(tile_dir, output_dir, overlap=32, pattern="*_tile_*.tif", marker=None, verbose=True):
    """
    Reconstruct all timepoints from a directory of tiles.
    
    Parameters:
    -----------
    tile_dir : str
        Directory containing tile files
    output_dir : str
        Directory to save reconstructed images
    overlap : int
        Overlap in pixels used during tiling (default: 32)
    pattern : str
        Glob pattern to match tile files
    marker : str, optional
        If provided, only process files containing this marker in their filename
        (e.g., 'cellprob', 'dP', 'magnitude')
    verbose : bool
        Enable verbose output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Group tiles by timepoint
    grouped_tiles = group_tiles_by_timepoint(tile_dir, pattern, marker=marker)
    
    if marker:
        print(f"Filtering tiles containing marker: '{marker}'")
    print(f"Found {len(grouped_tiles)} timepoint(s) to reconstruct")
    
    # Reconstruct each timepoint
    for basename in natsorted(grouped_tiles.keys()):
        tile_files = grouped_tiles[basename]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Reconstructing: {basename}")
            print(f"{'='*60}")
        
        # Reconstruct
        reconstructed = reconstruct_from_tiles(tile_files, overlap_xy=overlap, verbose=verbose)
        
        # Save
        output_path = os.path.join(output_dir, f"{basename}_reconstructed.tif")
        if verbose:
            print(f"Saving to: {output_path}")
            print(f"Shape: {reconstructed.shape}, dtype: {reconstructed.dtype}")
        
        tiff.imwrite(output_path, reconstructed)
        
        if verbose:
            print(f"Successfully reconstructed {basename}")
    
    print(f"\nReconstruction complete! Processed {len(grouped_tiles)} timepoint(s)")


def main():
    """CLI for reconstructing images from tiles."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Reconstruct full images from tiles created by image2tiles.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Reconstruct all tiles in a directory
  python tiles2images.py --tile_dir tiles/ --output_dir reconstructed/
  
  # Specify overlap used during tiling
  python tiles2images.py --tile_dir tiles/ --output_dir reconstructed/ --overlap 64
  
  # Use custom pattern to match specific tiles
  python tiles2images.py --tile_dir tiles/ --output_dir reconstructed/ --pattern "*timepoint*_tile_*.tif"
        '''
    )
    
    parser.add_argument('--tile_dir', type=str, required=True,
                       help='Directory containing tile files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save reconstructed images')
    parser.add_argument('--overlap', type=int, default=32,
                       help='Overlap in pixels used during tiling (default: 32)')
    parser.add_argument('--pattern', type=str, default="*_tile_*.tif",
                       help='Glob pattern to match tile files (default: *_tile_*.tif)')
    parser.add_argument('--marker', type=str, default=None,
                       help='Only process tiles containing this marker in filename (e.g., cellprob, dP, magnitude)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    reconstruct_directory(
        tile_dir=args.tile_dir,
        output_dir=args.output_dir,
        overlap=args.overlap,
        pattern=args.pattern,
        marker=args.marker,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
