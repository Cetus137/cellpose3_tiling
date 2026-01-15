
import numpy as np
from natsort import natsorted
import os
from pathlib import Path
import glob
import tifffile as tiff

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

    image_shape = image.shape
    print("Input image shape:", image_shape)

    #squeeze singleton dimensions
    image = np.squeeze(image)

    print("Squeezed image shape:", image.shape)

    if len(image.shape) == 3:
        # duplicate the image 3 times in the first dimension to simulate 3 views
        print("Duplicating image to have 3 views")
        image = np.stack([image, image, image], axis=0)  # shape now (3, z, y, x)

    
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

def tile_save_directory(input_dir , output_dir, file_index, tile_size=(256, 256, 256), overlap=32, verbose=True):
    """
    Tile a 3D image from the input directory, process each tile with Cellpose,
    and save the segmentation masks to the output directory.
    
    Parameters:
    -----------
    input_dir : str
        Path to input directory containing TIFF files
    output_dir : str
        Path to output directory for saving segmentation masks
    file_index : int
        Index of the specific file to process from the input directory
    tile_size : tuple
        Size of each tile (z, y, x). Default is (256, 256, 256)
    overlap : int
        Overlap in pixels for XY dimensions. Default is 32
    verbose : bool
        Whether to enable verbose output. Default is True
    """

    # Get list of all input files
    input_files = sorted(glob.glob(os.path.join(input_dir, "restored_timepoint_*.tif")))
    input_files = natsorted(input_files)
    
    if file_index < 0 or file_index >= len(input_files):
        raise IndexError("file_index out of range")
    
    input_file = input_files[file_index]
    basename = os.path.splitext(os.path.basename(input_file))[0]
    
    # Load image
    image = tiff.imread(input_file)
    
    if verbose:
        print(f"Processing file: {input_file} with shape {image.shape}")
    
    # Tile image into overlapping tiles
    tiles = tile_image_3d_3views(image, tile_size=tile_size, overlap_xy=overlap)
    
    if verbose:
        print(f"Generated {len(tiles)} tiles from the image")

    #now save the tiles in teh output directory

    for i, tile in enumerate(tiles):
        data = tile['data']
        z_start = tile['z_start']
        z_end = tile['z_end']
        y_start = tile['y_start']
        y_end = tile['y_end']
        x_start = tile['x_start']
        x_end = tile['x_end']

        tile_filename = f"{basename}_tile_{i:04d}_z{z_start}-{z_end}_y{y_start}-{y_end}_x{x_start}-{x_end}.tif"
        tile_path = os.path.join(output_dir, tile_filename)
        tiff.imwrite(tile_path, data)
    
    # Process
    return tiles

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
    
    parser.add_argument('--input_dir', type=str, help='Path to input resotored timelapse TIFF file')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--file_index', type=int, default=None, help='Index of specific file to process from input directory')
    parser.add_argument('--model', type=str, help='Path to pretrained Cellpose model')
    parser.add_argument('--tile_size', nargs=3, type=int, default=[256, 256, 256],
                       help='Tile size (z y x). Default: 256 256 256')
    parser.add_argument('--overlap', type=int, default=32,
                       help='XY overlap in pixels. Default: 32')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    tile_save_directory(
        input_dir = args.input_dir,
        output_dir = args.output_dir,
        file_index = args.file_index,
        tile_size = tuple(args.tile_size),
        overlap = args.overlap,
        verbose = args.verbose
    )