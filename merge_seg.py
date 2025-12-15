
import numpy as np
import argparse
import tifffile as tiff
from scipy.ndimage import distance_transform_edt, gaussian_filter, find_objects, binary_dilation, generate_binary_structure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label as sk_label
from skimage.graph import rag_boundary
from skimage.filters import sobel


def merge_touching_pairs_3d(masks: np.ndarray,
                            interface_threshold: float = 0.5,
                            connectivity: int = 1,
                            verbose: bool = False) -> np.ndarray:
    """
    Merge touching pairs if their interface is large relative to the smaller cell.
    Handles cells with multiple touching partners iteratively.
    Uses RAG to efficiently get both adjacency and interface sizes.

    Parameters
    ----------
    masks : np.ndarray
        3D labeled array (Z, Y, X) with input masks.
    interface_threshold : float
        Merge if interface_size / min_cell_size >= this threshold.
        Range [0, 1]. Higher values = more conservative (less merging).
    connectivity : int
        Connectivity for determining adjacency (1→6N, 2→18N, 3→26N).
    verbose : bool
        Print diagnostic information.

    Returns
    -------
    np.ndarray
        3D labeled array with merged masks.
    """
    if masks.ndim != 3:
        raise ValueError("Input `masks` must be a 3D array (Z, Y, X)")

    # Work with a copy
    result = np.copy(masks)

    # Keep iterating until no more merges
    iteration = 0
    while True:
        iteration += 1
        
        labels = np.unique(result)
        labels = labels[labels > 0]
        
        if len(labels) == 0:
            break
        
        if verbose:
            print(f"Iteration {iteration}: {len(labels)} labels")
        
        # Build RAG with boundary counts - automatically computes interface sizes
        edge_map = np.ones_like(result, dtype=float)
        rag = rag_boundary(result, edge_map, connectivity=connectivity)
        
        # Get touching pairs from RAG
        touching_pairs = [(u, v) for u, v in rag.edges() if 0 not in (u, v)]
        
        if verbose:
            print(f"  Found {len(touching_pairs)} touching pairs")
        
        if len(touching_pairs) == 0:
            break
        
        # Compute merge decisions for all pairs
        merge_decisions = []
        
        for label1, label2 in touching_pairs:
            # Get interface size from RAG boundary edge data
            interface_size = rag[label1][label2]['count']  # Number of boundary pixels
            
            # Compute perimeters: sum of all edge counts (including background if label 0 exists)
            perimeter1 = sum(data['count'] for neighbor, data in rag[label1].items())
            perimeter2 = sum(data['count'] for neighbor, data in rag[label2].items())
            min_perimeter = min(perimeter1, perimeter2)
            
            # Compute relative interface size compared to minimum perimeter
            relative_interface = interface_size / min_perimeter if min_perimeter > 0 else 0
            
            if verbose:
                print(f"    Pair ({label1}, {label2}): interface={interface_size}, min_perimeter={min_perimeter}, relative={relative_interface:.3f}")
            
            # Decide whether to merge
            if relative_interface >= interface_threshold:
                merge_decisions.append((label1, label2, relative_interface))
        
        if len(merge_decisions) == 0:
            if verbose:
                print(f"No merges in iteration {iteration}, stopping.")
            break
        
        # Sort by relative interface (strongest connections first)
        merge_decisions.sort(key=lambda x: x[2], reverse=True)
        
        # One merge per cell per iteration (conservative approach)
        # Track which cells have already been merged in this iteration
        merged_in_iteration = set()
        final_merges = []
        
        for label1, label2, rel_int in merge_decisions:
            # Only merge if neither cell has been merged yet in this iteration
            if label1 not in merged_in_iteration and label2 not in merged_in_iteration:
                final_merges.append((label1, label2, rel_int))
                merged_in_iteration.add(label1)
                merged_in_iteration.add(label2)
        
        if len(final_merges) == 0:
            if verbose:
                print(f"No valid merges in iteration {iteration}, stopping.")
            break
        
        if verbose:
            print(f"  Found {len(final_merges)} pairs to merge (from {len(merge_decisions)} candidates)")
        
        # Merge pairs: assign all instances of label2 to label1
        for label1, label2, rel_int in final_merges:
            # Only merge if both labels still exist (safety check)
            if label2 in result:
                result[result == label2] = label1
                if verbose:
                    print(f"    Merged {label2} → {label1} (interface ratio: {rel_int:.3f})")

    # Relabel consecutively
    final_labels = np.unique(result)
    final_labels = final_labels[final_labels > 0]

    relabeled = np.zeros_like(result, dtype=np.uint16)
    for new_label, old_label in enumerate(final_labels, start=1):
        relabeled[result == old_label] = new_label

    if verbose:
        print(f"\nFinal: {len(final_labels)} labels after merging")

    return relabeled.astype(np.uint16)


def merge_timelapse_4d(input_path: str,
                        output_dir: str,
                        interface_threshold: float = 0.5,
                        connectivity: int = 1,
                        t_list = None,
                        save = True,
                        verbose: bool = False) -> None:
    """
    Process a 4D timelapse (T, Z, Y, X) by merging touching pairs at each timepoint.

    Parameters
    ----------
    input_path : str
        Path to input 4D TIFF file with labeled masks (T, Z, Y, X).
    output_path : str
        Path to output 4D TIFF file for processed results.
    interface_threshold : float
        Merge if interface_size / min_cell_size >= this threshold.
        Range [0, 1]. Higher values = more conservative (less merging).
    connectivity : int
        Connectivity for labeling (1=6N, 2=18N, 3=26N).
    verbose : bool
        Print progress information.
    """
    # Load 4D timelapse
    if verbose:
        print(f"Loading 4D timelapse from: {input_path}")

    timelapse = tiff.imread(input_path)

    # Generate output filename
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Handle different input shapes
    if timelapse.ndim == 5:
        # Check if any dimension is 1 and squeeze it
        if 1 in timelapse.shape:
            timelapse = np.squeeze(timelapse)
            if verbose:
                print(f"Squeezed 5D input to shape: {timelapse.shape}")
    elif timelapse.ndim == 3:
        # Check that no dimensions are 1, then add time axis
        if 1 not in timelapse.shape:
            timelapse = timelapse[np.newaxis, ...]
            if verbose:
                print(f"Added time axis to 3D input, new shape: {timelapse.shape}")

    if timelapse.ndim != 4:
        raise ValueError(f"Expected 4D input (T, Z, Y, X), got shape {timelapse.shape}")

    n_timepoints = timelapse.shape[0]
    
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

    if verbose:
        print(f"Timelapse shape: {timelapse.shape}")
        print(f"Processing parameters:")
        print(f"  interface_threshold: {interface_threshold}")
        print(f"  connectivity: {connectivity}")

    # Process each frame
    result = []

    for idx, t in enumerate(timepoints_to_process):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing timepoint {t} ({idx+1}/{len(timepoints_to_process)})")
            print(f"{'='*60}")
        
        frame_masks = timelapse[t]
        
        frame_result = merge_touching_pairs_3d(
            frame_masks,
            interface_threshold=interface_threshold,
            connectivity=connectivity,
            verbose=verbose
        )
        
        result.append(frame_result)
        
        if verbose:
            n_labels = len(np.unique(frame_result[frame_result > 0]))
            print(f"  Frame {t+1}: {n_labels} labels")
    
    # Squeeze out single time dimension for single timepoint

    result_array = np.array(result, dtype=np.uint16)
    if result_array.shape[0] == 1:
        result_array = result_array[0]
    
    if save:
        # Create time range string
        if len(timepoints_to_process) == 0:
            raise ValueError("No timepoints to process")
        elif len(timepoints_to_process) == n_timepoints:
            time_str = "all"
        elif len(timepoints_to_process) == 1:
            time_str = f"t{timepoints_to_process[0]:04d}"
        else:
            time_str = f"t{timepoints_to_process[0]:04d}-t{timepoints_to_process[-1]:04d}"
        
        output_filename = f"merge_threshold{interface_threshold:.2f}_{time_str}.tif"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save result
        if verbose:
            print(f"Saving output to: {output_path}")

        tiff.imwrite(output_path, result_array)

    if verbose:
        print("Done!")
    return result_array
        
def batch_tif_merge(input_dir: str,
                    output_dir: str,
                    file_index: int = None,
                    t_list = None,
                    interface_threshold: float = 0.5,
                    connectivity: int = 1,
                    verbose: bool = False,
                    phrase: str = None) -> None:
    """
    Batch process all TIFF files in a directory for merging touching pairs.

    Parameters
    ----------
    input_directory : str
        Directory containing input TIFF files.      
        
    output_directory : str
        Directory to save output TIFF files.    
    interface_threshold : float
        Merge if interface_size / min_cell_size >= this threshold.
        Range [0, 1]. Higher values = more conservative (less merging).
    connectivity : int
        Connectivity for labeling (1=6N, 2=18N, 3=26N).
    verbose : bool
        Print progress information.
    """
    import os
    import glob
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
        output_path = output_dir / tif_file.replace('.tif', f'_merged_thresh{interface_threshold:.2f}.tif')
        if verbose:
            print(f"\nProcessing file: {tif_file}")
        
        results = merge_timelapse_4d(
            str(input_path),
            str(output_dir),
            interface_threshold=interface_threshold,
            connectivity=connectivity,
            verbose=verbose,
            save=False,
            t_list=t_list
        )

        if verbose:
            print( 'shape to be saved:', results.shape )

        tiff.imwrite(output_path, results)
        if verbose:
            print(f"Saved merged output to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='3D/4D watershed segmentation processing individual masks.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='Input directory containing TIFF files with 3D or 4D labeled masks')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Output directory for result TIFF file')
    parser.add_argument('--connectivity', '-c', type=int, default=1, choices=[1, 2, 3],
                        help='Connectivity for labeling (1=6N, 2=18N, 3=26N)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print progress information')
    parser.add_argument('--interface_threshold', '-t', type=float, default=0.5, 
                        help='Merge if interface_size / min_cell_size >= this threshold (0-1)') 
    parser.add_argument('--t_list', nargs='+', type=int, metavar='T', default=None,
                        help='Specific timepoints to process. E.g., --t_list 0 5 10 15. If not provided, all timepoints are processed.')
    parser.add_argument('--file_index', type=int, default=None,
                        help='Process only the file at this index (for parallel processing)')
                        
    
    args = parser.parse_args()
        
    # Print timepoint info
    if args.t_list is not None:
        print(f"Processing specific timepoints: {args.t_list}")
    else:
        print("Processing all timepoints")
    # Process timelapse (handles both 3D and 4D inputs)
    batch_tif_merge(
        args.input_dir,
        args.output_dir,
        interface_threshold=args.interface_threshold,
        connectivity=args.connectivity,
        t_list=args.t_list,
        file_index=args.file_index,
        verbose=args.verbose
    )
