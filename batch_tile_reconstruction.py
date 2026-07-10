import numpy as np
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
        # Filenames have TWO sets of coordinates:
        # 1. Crop coordinates: z53-309_y0-2048_x528-1552 (region from full volume)
        # 2. Tile coordinates: tile_0000_z0-256_y0-256_x0-256 (individual tile)
        # We need the TILE coordinates (after "tile_XXXX_")

        # Find all coordinate matches
        all_z = re.findall(r'_z(\d+)-(\d+)', flow_file)
        all_y = re.findall(r'_y(\d+)-(\d+)', flow_file)
        all_x = re.findall(r'_x(\d+)-(\d+)', flow_file)

        if len(all_z) < 1 or len(all_y) < 1 or len(all_x) < 1:
            print(f"Warning: Could not parse tile coordinates from {flow_file}, skipping.")
            continue

        # Use the LAST match (tile coordinates, not crop coordinates)
        z_start, z_end = int(all_z[-1][0]), int(all_z[-1][1])
        y_start, y_end = int(all_y[-1][0]), int(all_y[-1][1])
        x_start, x_end = int(all_x[-1][0]), int(all_x[-1][1])

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
                                 min_size=5000, max_size=None, diameter=None,
                                 cellprob_threshold=0.0, flow_threshold=0.4,
                                 do_3D=True, omni=False, verbose=True):
    dP, cellprob = timepoint_reconstruct_dP_cellprob(
        tile_dir, timepoint, overlap_xy
    )
    if verbose:
        print(f"Reconstructed dP shape: {dP.shape}, cellprob shape: {cellprob.shape}")

    if verbose:
        print("Computing masks from reconstructed dP and cellprob...")

    if omni:
        from omnipose.core import compute_masks as omni_compute_masks
        masks, _, _, _, _ = omni_compute_masks(
            dP, cellprob,
            min_size=min_size,
            flow_threshold=flow_threshold,
            do_3D=do_3D,
            mask_threshold=cellprob_threshold,
            cluster=False,
        )
    else:
        from cellpose.dynamics import compute_masks
        masks = compute_masks(dP,
                              cellprob,
                              min_size=min_size,
                              flow_threshold=flow_threshold,
                              do_3D=do_3D,
                              cellprob_threshold=cellprob_threshold)

    if max_size is not None:
        sizes = np.bincount(masks.ravel())
        large = np.where(sizes > max_size)[0]
        large = large[large != 0]  # keep background label 0
        if len(large):
            if verbose:
                print(f"Removing {len(large)} objects larger than max_size={max_size}")
            masks[np.isin(masks, large)] = 0

    if verbose:
        print(f"Reconstructed masks for timepoint {timepoint} with shape {masks.shape}")
        print(f"Unique labels in masks: {np.unique(masks)}")

    # Save masks
    output_path = os.path.join(output_dir, f'restored_timepoint_{timepoint:04d}_segmented.tif')
    print(f"Saving masks to {output_path}")
    tiff.imwrite(output_path, masks.astype(np.uint16))
    return


def reconstruct_single_channel(tiles, image_shape, overlap_xy=32):
    """
    Blend single-channel (Z, Y, X) tile data with linear-ramp overlap weighting.
    Each element of `tiles` is a dict with keys: data, z_start, z_end, y_start,
    y_end, x_start, x_end, original_shape.
    """
    z_size, y_size, x_size = image_shape
    vol    = np.zeros(image_shape, dtype=np.float32)
    weight = np.zeros(image_shape, dtype=np.float32)

    for tile_id, tile_info in enumerate(tiles):
        print('processing tile number', tile_id + 1)
        z_start = tile_info['z_start']
        z_end   = tile_info['z_end']
        y_start = tile_info['y_start']
        y_end   = tile_info['y_end']
        x_start = tile_info['x_start']
        x_end   = tile_info['x_end']
        actual_z, actual_y, actual_x = tile_info['original_shape']

        data = tile_info['data'][:actual_z, :actual_y, :actual_x]
        w = create_weight_map(
            (actual_z, actual_y, actual_x), overlap_xy,
            z_start, z_end, y_start, y_end, x_start, x_end,
            image_shape,
        )
        vol   [z_start:z_end, y_start:y_end, x_start:x_end] += data * w
        weight[z_start:z_end, y_start:y_end, x_start:x_end] += w

    return vol / np.maximum(weight, 1e-8)


def timepoint_reconstruct_boundary_fg(tile_dir, timepoint, overlap_xy=32,
                                      has_timepoints=True):
    """
    Load *_boundary.tif and *_fg.tif tiles for a timepoint, blend, and return
    (boundary, fg) as float32 arrays in [0, 1].
    """
    from natsort import natsorted
    import re

    tif_files = natsorted([f for f in os.listdir(tile_dir) if f.endswith('.tif')])

    if has_timepoints:
        boundary_files  = [f for f in tif_files
                           if f"timepoint_{timepoint:04d}" in f and f.endswith('_boundary.tif')]
        fg_files        = [f for f in tif_files
                           if f"timepoint_{timepoint:04d}" in f and f.endswith('_fg.tif')]
        centroid_files  = [f for f in tif_files
                           if f"timepoint_{timepoint:04d}" in f and f.endswith('_centroid.tif')]
    else:
        boundary_files  = [f for f in tif_files if f.endswith('_boundary.tif')]
        fg_files        = [f for f in tif_files if f.endswith('_fg.tif')]
        centroid_files  = [f for f in tif_files if f.endswith('_centroid.tif')]

    if not boundary_files:
        raise ValueError(f"No _boundary.tif files found for timepoint {timepoint} in {tile_dir}")
    if len(boundary_files) != len(fg_files):
        raise ValueError(
            f"Mismatch: {len(boundary_files)} boundary files vs {len(fg_files)} fg files"
        )
    has_centroid = len(centroid_files) == len(boundary_files)
    if centroid_files and not has_centroid:
        print(f"Warning: {len(centroid_files)} centroid files vs {len(boundary_files)} "
              f"boundary files — skipping centroid reconstruction")

    boundary_tiles, fg_tiles, centroid_tiles = [], [], []
    max_z = max_y = max_x = 0

    for idx, (b_file, fg_file) in enumerate(zip(boundary_files, fg_files)):
        all_z = re.findall(r'_z(\d+)-(\d+)', b_file)
        all_y = re.findall(r'_y(\d+)-(\d+)', b_file)
        all_x = re.findall(r'_x(\d+)-(\d+)', b_file)

        if not (all_z and all_y and all_x):
            print(f"Warning: cannot parse coordinates from {b_file}, skipping")
            continue

        z_start, z_end = int(all_z[-1][0]), int(all_z[-1][1])
        y_start, y_end = int(all_y[-1][0]), int(all_y[-1][1])
        x_start, x_end = int(all_x[-1][0]), int(all_x[-1][1])

        max_z = max(max_z, z_end)
        max_y = max(max_y, y_end)
        max_x = max(max_x, x_end)

        b_data  = tiff.imread(os.path.join(tile_dir, b_file)).astype(np.float32)
        fg_data = tiff.imread(os.path.join(tile_dir, fg_file)).astype(np.float32)
        if b_data.max()  > 1.5:
            b_data  /= 65535.0
        if fg_data.max() > 1.5:
            fg_data /= 65535.0

        tile_meta = dict(
            z_start=z_start, z_end=z_end,
            y_start=y_start, y_end=y_end,
            x_start=x_start, x_end=x_end,
            original_shape=(z_end - z_start, y_end - y_start, x_end - x_start),
        )
        boundary_tiles.append({**tile_meta, 'data': b_data})
        fg_tiles.append({**tile_meta,       'data': fg_data})

        if has_centroid:
            c_data = tiff.imread(
                os.path.join(tile_dir, centroid_files[idx])).astype(np.float32)
            if c_data.max() > 1.5:
                c_data /= 65535.0
            centroid_tiles.append({**tile_meta, 'data': c_data})

    image_shape = (max_z, max_y, max_x)
    print(f"Reconstructing {len(boundary_tiles)} tiles, image shape: {image_shape}")

    boundary = reconstruct_single_channel(boundary_tiles, image_shape, overlap_xy)
    fg       = reconstruct_single_channel(fg_tiles,       image_shape, overlap_xy)
    centroid = (reconstruct_single_channel(centroid_tiles, image_shape, overlap_xy)
                if has_centroid else None)
    return boundary, fg, centroid


def reconstruct_boundary_from_tiles(tile_dir, output_dir, timepoint, overlap_xy=32,
                                    has_timepoints=True, verbose=True):
    boundary, fg, centroid = timepoint_reconstruct_boundary_fg(
        tile_dir, timepoint, overlap_xy, has_timepoints=has_timepoints
    )
    if verbose:
        print(f"Boundary shape: {boundary.shape}  fg shape: {fg.shape}")

    os.makedirs(output_dir, exist_ok=True)
    b_path  = os.path.join(output_dir, f'restored_timepoint_{timepoint:04d}_boundary.tif')
    fg_path = os.path.join(output_dir, f'restored_timepoint_{timepoint:04d}_fg.tif')

    tiff.imwrite(b_path,  (boundary * 65535).astype(np.uint16), compression='zlib')
    print(f"Saved boundary: {b_path}")
    tiff.imwrite(fg_path, (fg * 65535).astype(np.uint16),       compression='zlib')
    print(f"Saved fg:       {fg_path}")

    # Interior map computed post-blending (no tile artefacts).
    interior      = (1.0 - boundary) * fg
    interior_path = os.path.join(output_dir, f'restored_timepoint_{timepoint:04d}_interior.tif')
    tiff.imwrite(interior_path, (interior * 65535).astype(np.uint16), compression='zlib')
    print(f"Saved interior: {interior_path}")

    if centroid is not None:
        c_path = os.path.join(output_dir, f'restored_timepoint_{timepoint:04d}_centroid.tif')
        tiff.imwrite(c_path, (centroid * 65535).astype(np.uint16), compression='zlib')
        print(f"Saved centroid: {c_path}")


def main():
    """
    CLI for reconstructing full images from tiles.
    """
    import argparse
    import distutils.util

    parser = argparse.ArgumentParser(description='Reconstruct images from tiled flow and cellprob outputs')
    parser.add_argument('--tile_dir',           type=str, required=True,
                        help='Directory containing the tile files')
    parser.add_argument('--output_dir',         type=str, required=True,
                        help='Directory to save reconstructed outputs')
    parser.add_argument('--timepoint',          type=int, required=True,
                        help='Timepoint number to reconstruct')
    parser.add_argument('--overlap',            type=int, default=32,
                        help='Overlap in pixels used during tiling (default: 32)')
    parser.add_argument('--min_size',           type=int, default=5000,
                        help='Minimum size of objects to keep (default: 5000)')
    parser.add_argument('--max_size',           type=int, default=None,
                        help='Maximum size of objects to keep; larger objects are removed (default: None)')
    parser.add_argument('--flow_threshold',     type=float, default=0.4,
                        help='Flow error threshold for compute_masks (default: 0.4)')
    parser.add_argument('--diameter',           type=str, default="None",
                        help='Cell diameter for Cellpose model (default: None)')
    parser.add_argument('--cellprob_threshold', type=float, default=0.0,
                        help='Cell probability threshold for segmentation (default: 0.0)')
    parser.add_argument('--do_3D',              type=str, default="True",
                        help='Whether to perform 3D segmentation (True/False)')
    parser.add_argument('--omni',               action='store_true', default=False,
                        help='Use omnipose compute_masks (for _dP_omni / _cellprob_omni tiles)')
    parser.add_argument('--boundary',           action='store_true', default=False,
                        help='Reconstruct affinity boundary + foreground maps instead of dP/cellprob')
    parser.add_argument('--no_timepoints',      action='store_true', default=False,
                        help='All tiles belong to one timepoint (no timepoint token in filename)')
    parser.add_argument('--verbose',            action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.boundary:
        reconstruct_boundary_from_tiles(
            tile_dir=args.tile_dir,
            output_dir=args.output_dir,
            timepoint=args.timepoint,
            overlap_xy=args.overlap,
            has_timepoints=not args.no_timepoints,
            verbose=args.verbose,
        )
    else:
        diameter = None if args.diameter == "None" else float(args.diameter)
        do_3D    = bool(distutils.util.strtobool(args.do_3D))
        reconstruct_masks_from_tiles(
            tile_dir=args.tile_dir,
            output_dir=args.output_dir,
            timepoint=args.timepoint,
            overlap_xy=args.overlap,
            min_size=args.min_size,
            max_size=args.max_size,
            diameter=diameter,
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold,
            do_3D=do_3D,
            omni=args.omni,
            verbose=args.verbose,
        )
    print("Reconstruction complete.")


if __name__ == "__main__":
    main()
