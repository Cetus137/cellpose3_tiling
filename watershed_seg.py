import numpy as np
import argparse
import tifffile as tiff
from scipy.ndimage import distance_transform_edt, gaussian_filter, find_objects, binary_dilation, generate_binary_structure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label as sk_label


def watershed_individual_masks_3d(masks: np.ndarray,
								  min_distance: int = 5,
								  connectivity: int = 1,
								  sigma: float = 1.0,
								  num_peaks_per_label: int = np.inf,
								  verbose: bool = False) -> np.ndarray:
	"""
	Apply watershed to each individual mask separately without fusion.
	Optimized using bounding boxes to process only relevant subvolumes.
	
	Parameters
	----------
	masks : np.ndarray
		3D labeled array (Z, Y, X) with input masks.
	min_distance : int
		Minimum distance between watershed seeds.
	connectivity : int
		Connectivity for labeling.
	sigma : float
		Gaussian blur sigma for distance transform smoothing.
	verbose : bool
		Print diagnostic information.
	
	Returns
	-------
	np.ndarray
		3D labeled array with watershed applied to each mask individually.
	"""
	if masks.ndim != 3:
		raise ValueError("Input `masks` must be a 3D array (Z, Y, X)")
	
	labels = np.unique(masks)
	labels = labels[labels > 0]
	
	if len(labels) == 0:
		return np.zeros_like(masks, dtype=np.uint16)
	
	if verbose:
		print(f"Processing {len(labels)} individual masks...")
	
	# Start with zeros - we'll assign new consecutive labels
	result = np.zeros_like(masks, dtype=np.uint16)
	next_label = 1
	
	# Precompute bounding boxes for all labels - this is very fast
	slices = find_objects(masks)
	
	# Process each label individually
	for idx, label in enumerate(labels):
		if verbose and (idx + 1) % 10 == 0:
			print(f"Processing mask {idx + 1}/{len(labels)}...")
		
		# Get bounding box for this label
		label_slice = slices[label - 1]
		if label_slice is None:
			continue
		
		# Extract subvolume containing this label (much smaller than full volume)
		subvolume = masks[label_slice]
		label_mask = (subvolume == label)
		
		# Skip if empty
		if not np.any(label_mask):
			continue
		
		# Compute distance transform on the small subvolume only
		distance = distance_transform_edt(label_mask)
		distance = gaussian_filter(distance, sigma=sigma)
		
		# Find peaks
		peak_idx = peak_local_max(
			distance,
			labels=label_mask.astype(np.uint8),
			min_distance=min_distance,
			exclude_border=True,
			footprint=None,
			num_peaks=num_peaks_per_label,
		)
		
		# Convert peaks to markers
		peaks = np.zeros_like(label_mask, dtype=bool)
		if len(peak_idx) > 0:
			peaks[tuple(peak_idx.T)] = True
		
		markers = sk_label(peaks, connectivity=connectivity)
		
		if markers.max() == 0:
			# No peaks found - keep original mask with new label
			result[label_slice][label_mask] = next_label
			next_label += 1
		else:
			# Run watershed on the small subvolume
			ws_result = watershed(-distance, markers=markers, mask=label_mask)
			
			# Relabel watershed result with consecutive global labels
			ws_labels = np.unique(ws_result[ws_result > 0])
			for ws_label in ws_labels:
				ws_mask = (ws_result == ws_label)
				result[label_slice][ws_mask] = next_label
				next_label += 1
	
	if verbose:
		final_count = len(np.unique(result[result > 0]))
		print(f"Watershed produced {final_count} total masks from {len(labels)} input masks")
	
	return result.astype(np.uint16)



def process_timelapse_4d(input_path: str,
						 output_dir: str,
						 min_distance: int = 5,
						 connectivity: int = 1,
						 sigma: float = 1.0,
						 num_peaks_per_label: int = np.inf,
						 t_list = None,
						 save = True,
						 verbose: bool = False) -> None:
	"""
	Process a 4D timelapse (T, Z, Y, X) by applying watershed to each timepoint.

	Parameters
	----------
	input_path : str
		Path to input 4D TIFF file with labeled masks (T, Z, Y, X).
	output_dir : str
		Directory to save output TIFF file.
	min_distance : int
		Minimum distance between watershed seeds (pixels).
	connectivity : int
		Connectivity for labeling (1=6N, 2=18N, 3=26N).
	sigma : float
		Gaussian blur sigma for distance transform smoothing.
	num_peaks_per_label : int
		Maximum number of peaks per label.
	t_range : tuple, list, or None
		Timepoints to process. Can be (start, end) tuple or list of specific timepoints.
	verbose : bool
		Print progress information.
	"""
	# Load 4D timelapse
	if verbose:
		print(f"Loading 4D timelapse from: {input_path}")

	timelapse = tiff.imread(input_path)

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
		print(f"Timelapse shape: {timelapse.shape} ({n_timepoints} frames)")
		print(f"Processing {len(timepoints_to_process)} timepoints: {timepoints_to_process}")
		print(f"Processing parameters:")
		print(f"  min_distance: {min_distance}")
		print(f"  connectivity: {connectivity}")
		print(f"  sigma: {sigma}")
	
	# Process each frame
	result = []
	
	for idx, t in enumerate(timepoints_to_process):
		if verbose:
			print(f"\nProcessing timepoint {t} ({idx+1}/{len(timepoints_to_process)})...")
		
		frame_masks = timelapse[t]
		
		frame_result = watershed_individual_masks_3d(
			frame_masks,
			min_distance=min_distance,
			connectivity=connectivity,
			sigma=sigma,
			verbose=verbose,
			num_peaks_per_label=num_peaks_per_label,
		)
		
		result.append(frame_result)
		
		if verbose:
			n_labels = len(np.unique(frame_result[frame_result > 0]))
			print(f"  Timepoint {t}: {n_labels} labels")

	# Save result
	result_array = np.array(result, dtype=np.uint16)

	# Squeeze out single time dimension for single timepoint
	if result_array.shape[0] == 1:
		result_array = result_array[0]
	
	if save:
		# Generate output filename
		import os
		os.makedirs(output_dir, exist_ok=True)
		
		# Create time range string
		if len(timepoints_to_process) == 0:
			raise ValueError("No timepoints to process")
		elif len(timepoints_to_process) == n_timepoints:
			time_str = "all"
		elif len(timepoints_to_process) == 1:
			time_str = f"t{timepoints_to_process[0]:04d}"
		else:
			time_str = f"t{timepoints_to_process[0]:04d}-t{timepoints_to_process[-1]:04d}"
		
		output_filename = f"watershed_{time_str}.tif"
		output_path = os.path.join(output_dir, output_filename)
	

		tiff.imwrite(output_path, result_array)


	if verbose:
		print("Done!")

	return result_array

def batch_tif_ws(input_dir: str,
					output_dir: str,
					file_index: int = None,
					num_peaks_per_label: int = np.inf,
					t_list = None,
					min_distance: int = 5,
					sigma: float = 1.0,
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
		output_path = output_dir / tif_file.replace('.tif', f'_ws.tif')
		if verbose:
			print(f"\nProcessing file: {tif_file}")
		
		results = process_timelapse_4d(
			str(input_path),
			str(output_dir),
			num_peaks_per_label=num_peaks_per_label,
			min_distance=min_distance,
			sigma=sigma,
			connectivity=connectivity,
			verbose=verbose,
			save=False,
			t_list=t_list
		)

		tiff.imwrite(output_path, results)
		if verbose:
			print(f"Saved ws output to: {output_path}")




if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description='3D/4D watershed segmentation processing individual masks.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	
	parser.add_argument('--input_dir', '-i', type=str, required=True,
						help='Input directory containing TIFF files with 3D or 4D labeled masks')
	parser.add_argument('--output_dir', '-o', type=str, required=True,
						help='Output directory for result TIFF file')
	parser.add_argument('--min_distance', '-d', type=int, default=5,
						help='Minimum distance between watershed seeds (pixels)')
	parser.add_argument('--connectivity', '-c', type=int, default=1, choices=[1, 2, 3],
						help='Connectivity for labeling (1=6N, 2=18N, 3=26N)')
	parser.add_argument('--sigma', '-s', type=float, default=1.0,
						help='Gaussian blur sigma for distance transform smoothing')
	parser.add_argument('--verbose', '-v', action='store_true',
						help='Print progress information')
	parser.add_argument('--num_peaks_per_label', '-n', type=int, default=np.inf,
						help='Maximum number of peaks per label for watershed seeds')
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
	batch_tif_ws(
		args.input_dir,
		args.output_dir,
		min_distance=args.min_distance,
		connectivity=args.connectivity,
		sigma=args.sigma,
		verbose=args.verbose,
		num_peaks_per_label=args.num_peaks_per_label,
		t_list=args.t_list,
		file_index=args.file_index
	)
