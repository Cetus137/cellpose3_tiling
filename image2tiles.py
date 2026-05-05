
import numpy as np
from natsort import natsorted
import os
from pathlib import Path
import glob
import tifffile as tiff

def tile_image_3d_3views(image, tile_size=(380, 256, 256), overlap_xy=32, timepoint=None):
    """
    Tile a 3D image into overlapping tiles.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input 3D image with shape (z, y, x) or (T, z, y, x) for timelapse
    tile_size : tuple
        Size of each tile (z, y, x). Default is (256, 256, 256)
    overlap_xy : int
        Overlap in pixels for XY dimensions. Default is 32
    timepoint : int or None
        If provided, indicates this is a single timepoint from a timelapse
        
    Returns:
    --------
    tiles : list of dict
        List of dictionaries containing:
            - 'data': the tile data
            - 'z_start', 'z_end': z coordinates
            - 'y_start', 'y_end': y coordinates  
            - 'x_start', 'x_end': x coordinates
            - 'timepoint': timepoint index (if applicable)
    """

    image_shape = image.shape
    print("Input image shape:", image_shape)

    #squeeze singleton dimensions
    image = np.squeeze(image)

    print("Squeezed image shape:", image.shape)

    if len(image.shape) == 3:
        # add single dimesnion for views
        print("Adding single dimension for 3 views")
        image = np.expand_dims(image, axis=0)  # shape becomes (1, z, y, x)

    
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
                
                tile_dict = {
                    'data': tile_data,
                    'z_start': z_start,
                    'z_end': z_end,
                    'y_start': y_start,
                    'y_end': y_end,
                    'x_start': x_start,
                    'x_end': x_end,
                    'original_shape': (z_end - z_start, y_end - y_start, x_end - x_start)
                }
                
                if timepoint is not None:
                    tile_dict['timepoint'] = timepoint
                
                tiles.append(tile_dict)

    return tiles

def tile_save_directory(output_dir, input_dir=None, file_path=None, tile_size=(380, 256, 256), overlap=32, verbose=True , phrase =None):
    """
    Tile a 3D image from the input directory, to the output directory.
    
    Parameters:
    -----------
    output_dir : str
        Path to output directory for saving segmentation masks
    input_dir : str or None
        Path to input directory containing TIFF files. Required if file_path is not provided.
    file_path : str or None
        Path to the specific file to process. If provided, input_dir is not needed.
    tile_size : tuple
        Size of each tile (z, y, x). Default is (256, 256, 256)
    overlap : int
        Overlap in pixels for XY dimensions. Default is 32
    verbose : bool
        Whether to enable verbose output. Default is True
    phrase : str or None
        Filename pattern to match when using input_dir (default: "*.tif*")
    """

    # Validate that either file_path or input_dir is provided
    if file_path is None and input_dir is None:
        raise ValueError("Either file_path or input_dir must be provided")

    if file_path is not None:
        # Use the provided file path directly
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        input_file = file_path
        basename = os.path.splitext(os.path.basename(input_file))[0]
        if verbose:
            print(f"Using provided file_path: {file_path}")
    else:
        # Search for files in input_dir using pattern
        print(f"DEBUG: phrase parameter = '{phrase}'")
        print(f"DEBUG: input_dir = '{input_dir}'")
        
        # Use default pattern if phrase is None
        if phrase is None:
            phrase = "*.tif*"
            print(f"DEBUG: Using default pattern")
        
        pattern = os.path.join(input_dir, phrase)
        print(f"DEBUG: Full glob pattern = '{pattern}'")
        
        input_files = glob.glob(pattern)
        input_files = natsorted(input_files)
        
        if verbose:
            print(f"Found files (first 3): {input_files[:3] if len(input_files) > 0 else 'NONE'}")
            
        print(f"Found {len(input_files)} input files in {input_dir}")
        
        if len(input_files) == 0:
            raise ValueError(f"No files found matching pattern: {pattern}")
            
        print("No file path provided, processing the first file matching the pattern")
        print("Available files:")
        for i, f in enumerate(input_files):
            print(f"  [{i}]: {f}")
        
        input_file = input_files[0]
        basename = os.path.splitext(os.path.basename(input_file))[0]
    
    # Load image with metadata to detect channel axis
    with tiff.TiffFile(input_file) as tif:
        image = tif.asarray()
        axes = tif.series[0].axes.upper() if tif.series else None

    if verbose:
        print(f"Processing file: {input_file} with shape {image.shape}, axes: {axes}")

    # Split channels: use OME-TIFF axis labels when available; for 5D data without
    # metadata fall back to TCZYX convention (channels at axis 1)
    if axes and 'C' in axes:
        c_axis = axes.index('C')
        num_channels = image.shape[c_axis]
        channel_images = [np.take(image, c, axis=c_axis) for c in range(num_channels)]
        if verbose:
            print(f"Detected {num_channels} channel(s) at axis {c_axis} from metadata (axes: {axes})")
    elif len(image.shape) == 5:
        c_axis = 1
        num_channels = image.shape[c_axis]
        channel_images = [np.take(image, c, axis=c_axis) for c in range(num_channels)]
        if verbose:
            print(f"No channel metadata; 5D array — assuming TCZYX, {num_channels} channel(s) at axis 1")
    else:
        num_channels = 1
        channel_images = [image]

    all_tiles = []

    for c_idx, ch_image in enumerate(channel_images):
        if verbose and num_channels > 1:
            print(f"Processing channel {c_idx + 1}/{num_channels}")

        ch_suffix = f"_ch{c_idx}" if num_channels > 1 else ""

        # Detect timelapse within this channel's data
        original_shape = ch_image.shape
        squeezed_shape = np.squeeze(ch_image).shape

        if verbose:
            print(f"Squeezed shape for detection: {squeezed_shape}")

        is_timelapse = False
        num_timepoints = 1

        if len(squeezed_shape) == 4:
            if squeezed_shape[0] > 3:
                is_timelapse = True
                num_timepoints = squeezed_shape[0]
                print(f"Detected timelapse data with {num_timepoints} timepoints")
                ch_image = np.squeeze(ch_image)
            else:
                print(f"Detected 4D data with {squeezed_shape[0]} views/slices (not timelapse)")
        elif len(original_shape) == 5:
            if original_shape[1] == 1 and original_shape[0] > 3:
                is_timelapse = True
                num_timepoints = original_shape[0]
                print(f"Detected 5D timelapse data with {num_timepoints} timepoints")
                ch_image = np.squeeze(ch_image)
            else:
                print(f"Detected 5D data (shape: {original_shape}) - treating as single volume")

        if is_timelapse:
            for t in range(num_timepoints):
                if verbose:
                    print(f"Processing timepoint {t+1}/{num_timepoints}")
                timepoint_image = ch_image[t]
                tiles = tile_image_3d_3views(timepoint_image, tile_size=tile_size, overlap_xy=overlap, timepoint=t)
                for i, tile in enumerate(tiles):
                    z_start, z_end = tile['z_start'], tile['z_end']
                    y_start, y_end = tile['y_start'], tile['y_end']
                    x_start, x_end = tile['x_start'], tile['x_end']
                    tile_filename = f"timepoint_{t:04d}_{basename}_tile_{i:04d}_z{z_start}-{z_end}_y{y_start}-{y_end}_x{x_start}-{x_end}{ch_suffix}.tif"
                    tiff.imwrite(os.path.join(output_dir, tile_filename), tile['data'])
                all_tiles.extend(tiles)
        else:
            tiles = tile_image_3d_3views(ch_image, tile_size=tile_size, overlap_xy=overlap)
            for i, tile in enumerate(tiles):
                z_start, z_end = tile['z_start'], tile['z_end']
                y_start, y_end = tile['y_start'], tile['y_end']
                x_start, x_end = tile['x_start'], tile['x_end']
                tile_filename = f"{basename}_tile_{i:04d}_z{z_start}-{z_end}_y{y_start}-{y_end}_x{x_start}-{x_end}{ch_suffix}.tif"
                tiff.imwrite(os.path.join(output_dir, tile_filename), tile['data'])
            all_tiles.extend(tiles)

    if verbose:
        print(f"Generated {len(all_tiles)} total tiles across {num_channels} channel(s)")

    return all_tiles

if __name__ == "__main__":

    import argparse

    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description='tiling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
                Examples:
                        '''
                )
    
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--input_dir', type=str, default=None, help='Path to input directory containing TIFF files (required if --file_path not provided)')
    parser.add_argument('--file_path', type=str, default=None, help='Path to specific file to process (alternative to --input_dir)')
    parser.add_argument('--model', type=str, help='Path to pretrained Cellpose model')
    parser.add_argument('--tile_size', nargs=3, type=int, default=[380, 256, 256],
                       help='Tile size (z y x). Default: 380 256 256')
    parser.add_argument('--overlap', type=int, default=32,
                       help='XY overlap in pixels. Default: 32')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--phrase', type=str, default="*.tif*",
                       help='Filename pattern to match input files (default: *.tif*)')
    
    args = parser.parse_args()
    
    # Validate that at least one input method is provided
    if args.file_path is None and args.input_dir is None:
        parser.error('Either --file_path or --input_dir must be provided')
    
    tile_save_directory(
        output_dir = args.output_dir,
        input_dir = args.input_dir,
        file_path = args.file_path,
        tile_size = tuple(args.tile_size),
        overlap = args.overlap,
        verbose = args.verbose,
        phrase = args.phrase
    )