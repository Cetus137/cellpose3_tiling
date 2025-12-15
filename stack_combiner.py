

import numpy as np
import tifffile as tiff

def combine_timepoint_files_3views(input_dir, output_path=None, phrase=None, verbose=True):
    """
    Read all TIF files in a directory and combine them into a single timelapse video.
    
    Each input frame should have shape (3, Z, Y, X) representing 3 views.
    Output will have shape (3, T, Z, Y, X) where T is the number of timepoints.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the individual timepoint TIF files
    output_path : str or None
        Path for the output combined video file. If None, will save as 'combined_timelapse.tif' in input_dir
    phrase : str or None
        Optional substring that must be present in the filename. If provided, only files whose
        filenames contain this phrase (case-sensitive) will be included.
    verbose : bool
        Print progress information
        
    Returns:
    --------
    combined_video : numpy.ndarray
        Combined video with shape (3, T, Z, Y, X)
    output_path : str
        Path where the video was saved
    """
    import glob
    from pathlib import Path
    
    input_dir = Path(input_dir)
    input_name = input_dir.name
    output_name = input_name + "_combined_timelapse.tif"

    # Find all TIF files
    tif_files = sorted(glob.glob(str(input_dir / "*.tif")))

    # If phrase provided, filter filenames that contain the phrase
    if phrase is not None:
        if verbose:
            print(f"Filtering files for phrase: '{phrase}'")
        tif_files = [p for p in tif_files if phrase in Path(p).name]

    if len(tif_files) == 0:
        raise ValueError(f"No TIF files found in {input_dir}")

    if verbose:
        print(f"Found {len(tif_files)} TIF files in {input_dir}")

    # Find compatible files and determine spatial shape from first compatible file
    compatible_files = []
    Z = Y = X = None
    out_dtype = None
    
    for p in tif_files:
        if verbose:
            print(f"  Probing {Path(p).name}...")
        try:
            frame = tiff.imread(p)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not read {Path(p).name}: {e}")
            continue

        if frame.ndim == 4 and frame.shape[0] == 3:
            # first compatible: record shape
            if Z is None:
                _, Z, Y, X = frame.shape
                out_dtype = frame.dtype
                compatible_files.append(p)
            else:
                if frame.shape[1:] == (Z, Y, X):
                    compatible_files.append(p)
                else:
                    if verbose:
                        print(f"  Skipping {Path(p).name}: incompatible shape {frame.shape}, expected (3,{Z},{Y},{X})")
        else:
            if verbose:
                print(f"  Skipping {Path(p).name}: expected shape (3,Z,Y,X), got {frame.shape}")

    if len(compatible_files) == 0:
        raise ValueError("No valid frames found with shape (3, Z, Y, X)")

    T = len(compatible_files)
    if verbose:
        print(f"Using {T} compatible frame(s); spatial shape (Z,Y,X)=({Z},{Y},{X})")

    # Preallocate output array and fill per-file to avoid extra copies
    combined_video = np.empty((3, T, Z, Y, X), dtype=out_dtype)
    for t_idx, p in enumerate(compatible_files):
        if verbose:
            print(f"  Loading and assigning {Path(p).name} -> time index {t_idx}")
        frame = tiff.imread(p)
        combined_video[:, t_idx, ...] = frame

    if verbose:
        print(f"Combined video shape: {combined_video.shape}")

    # Save combined video (cast to float32 to preserve compatibility)
    # Generate output path if not provided
    if output_path is None:
        if phrase is not None:
            output_path = input_dir / f"combined_timelapse_{phrase}.tif"
        else:
            output_path = input_dir / "combined_timelapse.tif"

    else:
        if phrase is not None:
            output_path = Path(output_path)
            output_path = output_path / f"combined_timelapse_{phrase}.tif"
        else:
            output_path = Path(output_path)
            output_path = output_path / "combined_timelapse.tif"

    if verbose:
        print(f"Saving combined video to {output_path}")
    tiff.imwrite(str(output_path), combined_video.astype(np.float32))
    
    if verbose:
        print("Done!")
    
    return combined_video, str(output_path)

def combine_timepoint_files(input_dir, output_path=None, phrase=None, verbose=True, dtype=np.int16):
    """
    Read all TIF files in a directory and combine them into a single timelapse video.
    
    Each input frame should have shape (Z, Y, X) representing 3 views.
    Output will have shape (T, Z, Y, X) where T is the number of timepoints.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the individual timepoint TIF files
    output_path : str or None
        Path for the output combined video file. If None, will save as 'combined_timelapse.tif' in input_dir
    phrase : str or None
        Optional substring that must be present in the filename. If provided, only files whose
        filenames contain this phrase (case-sensitive) will be included.
    verbose : bool
        Print progress information
        
    Returns:
    --------
    combined_video : numpy.ndarray
        Combined video with shape (T, Z, Y, X)
    output_path : str
        Path where the video was saved
    """
    import glob
    from pathlib import Path
    
    input_dir = Path(input_dir)
    
    # Find all TIF files
    tif_files = sorted(glob.glob(str(input_dir / "*.tif")))

    # If phrase provided, filter filenames that contain the phrase
    if phrase is not None:
        if verbose:
            print(f"Filtering files for phrase: '{phrase}'")
        tif_files = [p for p in tif_files if phrase in Path(p).name]
    
    if len(tif_files) == 0:
        raise ValueError(f"No TIF files found in {input_dir}")
    
    if verbose:
        print(f"Found {len(tif_files)} TIF files in {input_dir}")
    
    # Load all timepoints
    frames = []
    for file_path in tif_files:
        if verbose:
            print(f"  Loading {Path(file_path).name}...")
        frame = tiff.imread(file_path)
        
        # Validate shape
        if frame.ndim == 4 and frame.shape[0] == 1:
            # Assume single T view stored as (1, Z, Y, X)
            frame = frame[0, ...]

        if frame.ndim != 3:
            if verbose:
                print(f"  Warning: Skipping {Path(file_path).name} - expected shape (Z, Y, X), got {frame.shape}")
            continue
            
        frames.append(frame)
    
    if len(frames) == 0:
        raise ValueError("No valid frames found with shape (Z, Y, X)")
    
    # Stack into video: (T, Z, Y, X)
    combined_video = np.stack(frames, axis=0)  # (T, Z, Y, X)
    
    if verbose:
        print(f"Combined video shape: {combined_video.shape}")
    
    # Generate output path if not provided
    if output_path is None:
        if phrase is not None:
            output_path = input_dir / f"combined_timelapse_{phrase}.tif"
        else:
            output_path = input_dir / "combined_timelapse.tif"

    else:
        if phrase is not None:
            output_path = Path(output_path)
            output_path = output_path / f"combined_timelapse_{phrase}.tif"
        else:
            output_path = Path(output_path)
            output_path = output_path / "combined_timelapse.tif"

    if dtype is not None:
        print(f"Converting combined video to dtype: {dtype}")
        combined_video = combined_video.astype(dtype)

    
    # Save combined video
    if verbose:
        print(f"Saving combined video to {output_path}")
    tiff.imwrite(str(output_path), combined_video)
    
    if verbose:
        print("Done!")
    
    return combined_video, str(output_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine individual timepoint TIF files into a single timelapse video.")
    
    parser.add_argument("--input_dir"   , type=str              , help="Directory containing individual timepoint TIF files.")
    parser.add_argument("--output_path" , type=str, default=None, help="Path to save the combined video. Defaults to 'combined_timelapse.tif' in input_dir.")
    parser.add_argument("--phrase"      , type=str, default=None, help="Optional substring to filter filenames.")
    parser.add_argument("--three_views" , action="store_true" , help="Indicates that input files have 3 views (shape: 3,Z,Y,X).")
    parser.add_argument("--verbose"     , action="store_true"   , help="Print progress information.")
    parser.add_argument("--dtype"       , type=str, default="int16", help="Data type for output video (e.g., 'int16', 'float32').")
    
    args = parser.parse_args()
    print()
    if args.three_views:
        #combine_timepoint_files_3views(args.input_dir, args.output_path, args.phrase, args.verbose)
        combine_timepoint_files_3views(args.input_dir, args.output_path, args.phrase, args.verbose)
    elif not args.three_views:
        #combine_timepoint_files(args.input_dir, args.output_path, args.phrase, args.verbose)
        combine_timepoint_files(args.input_dir, args.output_path, args.phrase, args.verbose , dtype=args.dtype)


