import numpy as np
import tifffile as tiff
from pathlib import Path
from scipy import ndimage
from collections import defaultdict
import glob
from difflib import SequenceMatcher
import re


def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    
    Parameters:
    -----------
    mask1, mask2 : numpy.ndarray
        Binary masks to compare
        
    Returns:
    --------
    iou : float
        IoU score between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_pixel_iou(seg_gt, seg_pred):
    """
    Calculate pixel-level IoU between two segmentations.
    Treats all non-zero pixels as foreground, regardless of label.
    
    Parameters:
    -----------
    seg_gt : numpy.ndarray
        Ground truth segmentation
    seg_pred : numpy.ndarray
        Predicted segmentation
        
    Returns:
    --------
    iou : float
        Pixel-level IoU score between 0 and 1
    """
    # Convert to binary masks (foreground vs background)
    mask_gt = seg_gt > 0
    mask_pred = seg_pred > 0
    
    return calculate_iou(mask_gt, mask_pred)


def calculate_pixel_dice(seg_gt, seg_pred):
    """
    Calculate pixel-level DICE score between two segmentations.
    Treats all non-zero pixels as foreground, regardless of label.
    
    DICE = 2 * |A ∩ B| / (|A| + |B|)
    
    Parameters:
    -----------
    seg_gt : numpy.ndarray
        Ground truth segmentation
    seg_pred : numpy.ndarray
        Predicted segmentation
        
    Returns:
    --------
    dice : float
        Pixel-level DICE score between 0 and 1
    """
    # Convert to binary masks (foreground vs background)
    mask_gt = seg_gt > 0
    mask_pred = seg_pred > 0
    
    intersection = np.logical_and(mask_gt, mask_pred).sum()
    size_gt = mask_gt.sum()
    size_pred = mask_pred.sum()
    
    if size_gt + size_pred == 0:
        return 1.0  # Both empty
    
    return 2 * intersection / (size_gt + size_pred)


def filter_by_depth(seg, max_depth, depth_axis=0):
    """
    Remove labeled objects that sit deeper than a cutoff along the depth axis.

    An object is *kept* only if the centroid of its voxels along ``depth_axis``
    is strictly less than ``max_depth`` (i.e. it lies in the shallow region of
    the stack). Every other object is set to background (0). This is used to
    restrict segmentation comparison to cells located above a given imaging
    depth, where image quality is typically higher.

    Parameters
    ----------
    seg : numpy.ndarray
        Labeled segmentation (0 = background).
    max_depth : int or float or None
        Depth cutoff in pixels. Objects whose centroid depth < max_depth are
        kept. If None, the segmentation is returned unchanged.
    depth_axis : int
        Axis corresponding to imaging depth (z). Default 0 for (Z, Y, X) volumes.

    Returns
    -------
    filtered : numpy.ndarray
        Copy of seg with objects deeper than the cutoff removed.
    """
    if max_depth is None:
        return seg

    flat = seg.ravel()
    counts = np.bincount(flat)

    # Only background present -> nothing to keep
    if counts.size <= 1:
        return seg

    # Depth coordinate of every voxel, broadcast along the depth axis
    nz = seg.shape[depth_axis]
    coord_shape = [1] * seg.ndim
    coord_shape[depth_axis] = nz
    depth_coord = np.arange(nz, dtype=np.int32).reshape(coord_shape)
    depth_coord = np.broadcast_to(depth_coord, seg.shape)

    # Per-label centroid depth = (sum of z over voxels) / (voxel count)
    depth_sum = np.bincount(flat, weights=depth_coord.ravel(), minlength=counts.size)
    with np.errstate(invalid='ignore', divide='ignore'):
        centroid_depth = depth_sum / counts

    keep = centroid_depth < max_depth
    keep[0] = False  # background is never an object

    # Look-up table maps kept labels to themselves, removed labels to 0
    lut = np.where(keep, np.arange(counts.size), 0).astype(seg.dtype, copy=False)
    return lut[seg]


def match_objects(seg_gt, seg_pred, iou_threshold=0.5):
    """
    Match objects between ground truth and prediction segmentations based on IoU.
    
    Parameters:
    -----------
    seg_gt : numpy.ndarray
        Ground truth segmentation with labeled objects (0 = background)
    seg_pred : numpy.ndarray
        Predicted segmentation with labeled objects (0 = background)
    iou_threshold : float
        Minimum IoU to consider objects as matching. Default is 0.5
        
    Returns:
    --------
    matches : dict
        Dictionary mapping ground truth label to predicted label
    unmatched_gt : set
        Set of ground truth labels without matches
    unmatched_pred : set
        Set of predicted labels without matches

    Notes:
    ------
    Vectorized implementation: the full overlap (contingency) table between GT
    and predicted labels is computed in a single pass, so every pairwise IoU is
    derived at once instead of rescanning the volume per object. The greedy
    assignment then reproduces the original semantics exactly -- GT labels are
    processed in ascending order, each taking its highest-IoU prediction that is
    still available and only if that IoU >= iou_threshold.
    """
    gt_flat = seg_gt.ravel()
    pred_flat = seg_pred.ravel()

    gt_labels = np.unique(gt_flat)
    pred_labels = np.unique(pred_flat)
    gt_labels = gt_labels[gt_labels != 0]
    pred_labels = pred_labels[pred_labels != 0]

    # Nothing to match if either side has no objects
    if gt_labels.size == 0 or pred_labels.size == 0:
        return {}, set(gt_labels.tolist()), set(pred_labels.tolist())

    # Per-label voxel counts (indexed by label value)
    gt_area = np.bincount(gt_flat)
    pred_area = np.bincount(pred_flat)

    # Overlap table over foreground-of-both voxels only, encoded as a single
    # key (gt_label * stride + pred_label) so np.unique gives per-pair counts
    # without ever allocating a dense gt-by-pred array.
    fg = (gt_flat != 0) & (pred_flat != 0)
    gi = gt_flat[fg].astype(np.int64)
    pj = pred_flat[fg].astype(np.int64)

    matches = {}
    matched_pred = set()

    if gi.size > 0:
        stride = np.int64(pred_area.size)
        keys, inter = np.unique(gi * stride + pj, return_counts=True)
        gt_of_pair = keys // stride
        pred_of_pair = keys % stride

        # IoU for every overlapping pair at once
        union = gt_area[gt_of_pair] + pred_area[pred_of_pair] - inter
        iou = inter / union

        # Keep only candidate pairs that meet the threshold, then assign greedily
        keep = iou >= iou_threshold
        cand_gt = gt_of_pair[keep]
        cand_pred = pred_of_pair[keep]
        cand_iou = iou[keep]

        # Order by GT ascending, then IoU descending, so the first unused pred
        # encountered for each GT is its best available match.
        order = np.lexsort((-cand_iou, cand_gt))
        for k in order:
            g = int(cand_gt[k])
            if g in matches:
                continue
            p = int(cand_pred[k])
            if p in matched_pred:
                continue
            matches[g] = p
            matched_pred.add(p)

    # Identify unmatched objects
    unmatched_gt = set(gt_labels.tolist()) - set(matches.keys())
    unmatched_pred = set(pred_labels.tolist()) - matched_pred

    return matches, unmatched_gt, unmatched_pred


def calculate_object_f1(seg_gt, seg_pred, iou_threshold=0.5, return_detailed=False,
                        max_depth=None, depth_axis=0):
    """
    Calculate object-level F1 score between two volumetric segmentations.
    
    Object-level F1 score:
    - Each labeled object in ground truth is matched to objects in prediction
    - Match is defined by IoU > threshold
    - TP: Ground truth objects with matching prediction
    - FP: Predicted objects without matching ground truth
    - FN: Ground truth objects without matching prediction
    - F1 = 2*TP / (2*TP + FP + FN)
    
    Parameters:
    -----------
    seg_gt : numpy.ndarray
        Ground truth segmentation with labeled objects (0 = background)
    seg_pred : numpy.ndarray
        Predicted segmentation with labeled objects (0 = background)
    iou_threshold : float
        Minimum IoU to consider objects as matching. Default is 0.5
    return_detailed : bool
        If True, return dictionary with detailed metrics
    max_depth : int or float or None
        If set, only cells whose centroid along depth_axis is < max_depth are
        included in the comparison (both GT and prediction are filtered).
    depth_axis : int
        Axis corresponding to imaging depth (z). Default 0 for (Z, Y, X) volumes.

    Returns:
    --------
    f1_score : float
        Object-level F1 score between 0 and 1
    or
    metrics : dict
        Dictionary containing f1_score, precision, recall, pixel_iou, pixel_dice, tp, fp, fn
    """
    # Restrict comparison to cells above the depth cutoff, if requested
    if max_depth is not None:
        seg_gt = filter_by_depth(seg_gt, max_depth, depth_axis)
        seg_pred = filter_by_depth(seg_pred, max_depth, depth_axis)

    # Match objects between segmentations
    matches, unmatched_gt, unmatched_pred = match_objects(seg_gt, seg_pred, iou_threshold)
    
    # Calculate object-level metrics
    tp = len(matches)  # True positives: matched ground truth objects
    fp = len(unmatched_pred)  # False positives: unmatched predicted objects
    fn = len(unmatched_gt)  # False negatives: unmatched ground truth objects
    
    # Calculate object-level F1 score
    if tp == 0 and fp == 0 and fn == 0:
        # Both segmentations are empty
        f1_score = 1.0
        precision = 1.0
        recall = 1.0
    elif tp == 0:
        # No matches
        f1_score = 0.0
        precision = 0.0
        recall = 0.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * tp / (2 * tp + fp + fn)
    
    # Calculate pixel-level metrics
    pixel_iou = calculate_pixel_iou(seg_gt, seg_pred)
    pixel_dice = calculate_pixel_dice(seg_gt, seg_pred)
    
    if return_detailed:
        return {
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
            'pixel_iou': pixel_iou,
            'pixel_dice': pixel_dice,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'n_gt_objects': tp + fn,
            'n_pred_objects': tp + fp
        }
    
    return f1_score


def compare_segmentation_pair(file_gt, file_pred, iou_threshold=0.5, return_detailed=False,
                              max_depth=None, depth_axis=0):
    """
    Load two volumetric tif segmentations and calculate object-level F1 score.
    
    Parameters:
    -----------
    file_gt : str or Path
        Path to ground truth segmentation tif file
    file_pred : str or Path
        Path to predicted segmentation tif file
    iou_threshold : float
        Minimum IoU to consider objects as matching. Default is 0.5
    return_detailed : bool
        If True, return dictionary with detailed metrics
        
    Returns:
    --------
    f1_score : float or dict
        Object-level F1 score (or detailed metrics if return_detailed=True)
    """
    # Load segmentation files
    print(f"Loading ground truth: {file_gt}")
    seg_gt = tiff.imread(file_gt)
    
    print(f"Loading prediction: {file_pred}")
    seg_pred = tiff.imread(file_pred)
    
    # Check shapes match
    if seg_gt.shape != seg_pred.shape:
        raise ValueError(f"Shape mismatch: GT {seg_gt.shape} vs Pred {seg_pred.shape}")
    
    print(f"Segmentation shape: {seg_gt.shape}")
    if max_depth is not None:
        print(f"Depth filter: keeping cells with centroid depth < {max_depth} px (axis {depth_axis})")

    # Calculate F1 score
    result = calculate_object_f1(seg_gt, seg_pred, iou_threshold, return_detailed,
                                 max_depth=max_depth, depth_axis=depth_axis)
    
    if return_detailed:
        print(f"Object-level F1 Score: {result['f1_score']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"Pixel-level IoU: {result['pixel_iou']:.4f}")
        print(f"Pixel-level DICE: {result['pixel_dice']:.4f}")
        print(f"TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}")
        print(f"GT objects: {result['n_gt_objects']}, Pred objects: {result['n_pred_objects']}")
    else:
        print(f"F1 Score: {result:.4f}")
    
    return result


def string_similarity(str1, str2):
    """
    Calculate string similarity ratio between 0 and 1.
    Higher score means more similar strings.
    """
    return SequenceMatcher(None, str1, str2).ratio()


def find_matching_pairs(dir_gt, dir_pred, pattern_gt="_masks.tif", pattern_pred="_restored_masks.tif",
                        match_by="position"):
    """
    Find matching pairs of segmentation files in two directories.
    
    Parameters:
    -----------
    dir_gt : str or Path
        Directory containing ground truth segmentations
    dir_pred : str or Path
        Directory containing predicted segmentations
    pattern_gt : str
        File pattern for ground truth files. Can be:
        - Glob pattern (e.g., "*_masks.tif", "*.tif") for finding files
        - Suffix string (e.g., "_masks.tif") for basename matching
        Default is "_masks.tif"
    pattern_pred : str
        File pattern for predicted files. Same format as pattern_gt.
        Default is "_restored_masks.tif"
    match_by : str
        Matching strategy: 
        - "position": Match by sorted position (most reproducible)
                     Patterns can include wildcards for file finding.
        - "similarity": Match by finding most similar filename (best for different names)
                       Uses string similarity to find closest match for each GT file.
        - "basename": Match by replacing pattern_gt with pattern_pred in filename
                     Patterns should NOT include wildcards - use literal suffixes.
        
    Returns:
    --------
    pairs : list of tuple
        List of (gt_file, pred_file) tuples
    """
    dir_gt = Path(dir_gt)
    dir_pred = Path(dir_pred)
    
    # Ensure patterns are valid glob patterns (don't double up on *)
    gt_glob = pattern_gt if pattern_gt.startswith('*') else f"*{pattern_gt}"
    pred_glob = pattern_pred if pattern_pred.startswith('*') else f"*{pattern_pred}"
    
    # Get all files matching the patterns
    gt_files = sorted(dir_gt.glob(gt_glob))
    pred_files = sorted(dir_pred.glob(pred_glob))
    
    pairs = []
    
    if match_by == "position":
        # Simple position-based matching
        if len(gt_files) != len(pred_files):
            print(f"WARNING: Different number of files in directories!")
            print(f"  GT: {len(gt_files)} files, Pred: {len(pred_files)} files")
            print(f"  Will match first {min(len(gt_files), len(pred_files))} pairs")
        
        for i, (gt_file, pred_file) in enumerate(zip(gt_files, pred_files)):
            pairs.append((gt_file, pred_file))
            print(f"Pair {i+1}: {gt_file.name} <-> {pred_file.name}")
    
    elif match_by == "similarity":
        # String similarity matching - find closest filename for each GT file
        used_pred_files = set()
        
        for gt_file in gt_files:
            best_match = None
            best_score = 0.0
            
            # Find the most similar prediction file
            for pred_file in pred_files:
                if pred_file in used_pred_files:
                    continue
                    
                similarity = string_similarity(gt_file.name, pred_file.name)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = pred_file
            
            if best_match is not None:
                pairs.append((gt_file, best_match))
                used_pred_files.add(best_match)
                print(f"Pair {len(pairs)}: {gt_file.name} <-> {best_match.name} (similarity: {best_score:.3f})")
            else:
                print(f"Warning: No matching prediction for {gt_file.name}")
        
        # Report any unmatched files
        if len(gt_files) != len(pairs):
            print(f"\\nWARNING: {len(gt_files) - len(pairs)} GT files could not be matched!")
        if len(pred_files) != len(pairs):
            print(f"WARNING: {len(pred_files) - len(pairs)} prediction files were not used!")
    
    elif match_by == "basename":
        # Original basename matching
        for gt_file in gt_files:
            # Extract base name by removing the pattern
            base_name = gt_file.name.replace(pattern_gt, "")
            
            # Construct expected prediction file name
            pred_file = dir_pred / f"{base_name}{pattern_pred}"
            
            if pred_file.exists():
                pairs.append((gt_file, pred_file))
                print(f"Found pair: {gt_file.name} <-> {pred_file.name}")
            else:
                print(f"Warning: No matching prediction for {gt_file.name}")
    
    else:
        raise ValueError(f"Invalid match_by value: {match_by}. Use 'position', 'similarity', or 'basename'")
    
    print(f"\nFound {len(pairs)} matching pairs")
    return pairs


def batch_compare_segmentations(dir_gt, dir_pred, pattern_gt="_masks.tif",
                                 pattern_pred="_restored_masks.tif",
                                 iou_threshold=0.5,
                                 save_results=None,
                                 match_by="position",
                                 max_depth=None,
                                 depth_axis=0):
    """
    Compare all matching segmentation pairs in two directories and calculate overall F1 score.
    
    Parameters:
    -----------
    dir_gt : str or Path
        Directory containing ground truth segmentations
    dir_pred : str or Path
        Directory containing predicted segmentations
    pattern_gt : str
        File pattern for ground truth files. Default is "_masks.tif"
    pattern_pred : str
        File pattern for predicted files. Default is "_restored_masks.tif"
    iou_threshold : float
        Minimum IoU to consider objects as matching. Default is 0.5
    save_results : str or Path or None
        If provided, save detailed results to this file
    match_by : str
        Matching strategy: "position" (default), "similarity" (best for mixed files), or "basename"
    max_depth : int or float or None
        If set, only cells whose centroid along depth_axis is < max_depth are
        included in the comparison (applied to both GT and prediction before
        matching, so all object- and pixel-level metrics reflect the filter).
    depth_axis : int
        Axis corresponding to imaging depth (z). Default 0 for (Z, Y, X) volumes.

    Returns:
    --------
    results : dict
        Dictionary containing:
            - 'overall_f1': Object-level F1 score aggregated across all pairs
            - 'overall_precision': Object-level precision aggregated
            - 'overall_recall': Object-level recall aggregated
            - 'overall_pixel_iou': Pixel-level IoU aggregated across all pixels
            - 'overall_pixel_dice': Pixel-level DICE aggregated across all pixels
            - 'mean_f1': Mean object-level F1 score across pairs
            - 'mean_pixel_iou': Mean pixel-level IoU across pairs
            - 'mean_pixel_dice': Mean pixel-level DICE across pairs
            - 'per_file_results': List of per-file results
            - 'total_tp', 'total_fp', 'total_fn': Aggregated object counts
    """
    # Find matching pairs
    pairs = find_matching_pairs(dir_gt, dir_pred, pattern_gt, pattern_pred, match_by)
    
    if len(pairs) == 0:
        print("No matching pairs found!")
        return None
    
    # Process each pair
    print("\n" + "="*80)
    print("Processing pairs...")
    print("="*80)
    
    per_file_results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pixel_intersection = 0
    total_pixel_union = 0
    total_pixel_gt = 0
    total_pixel_pred = 0
    
    for i, (gt_file, pred_file) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] Processing: {gt_file.name}")
        
        # Load and compare
        seg_gt = tiff.imread(gt_file)
        seg_pred = tiff.imread(pred_file)

        # Apply depth filter once here so that object- and pixel-level metrics
        # (accumulated below) are all computed on the same filtered volumes
        if max_depth is not None:
            seg_gt = filter_by_depth(seg_gt, max_depth, depth_axis)
            seg_pred = filter_by_depth(seg_pred, max_depth, depth_axis)

        # Calculate metrics (arrays already filtered, so pass max_depth=None)
        metrics = calculate_object_f1(seg_gt, seg_pred, iou_threshold, return_detailed=True)
        
        # Store results
        result_entry = {
            'gt_file': str(gt_file),
            'pred_file': str(pred_file),
            'f1_score': metrics['f1_score'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'pixel_iou': metrics['pixel_iou'],
            'pixel_dice': metrics['pixel_dice'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'n_gt_objects': metrics['n_gt_objects'],
            'n_pred_objects': metrics['n_pred_objects']
        }
        per_file_results.append(result_entry)
        
        # Accumulate totals
        total_tp += metrics['tp']
        total_fp += metrics['fp']
        total_fn += metrics['fn']
        
        # Accumulate pixel-level metrics
        mask_gt = seg_gt > 0
        mask_pred = seg_pred > 0
        total_pixel_intersection += np.logical_and(mask_gt, mask_pred).sum()
        total_pixel_union += np.logical_or(mask_gt, mask_pred).sum()
        total_pixel_gt += mask_gt.sum()
        total_pixel_pred += mask_pred.sum()
        
        print(f"  F1: {metrics['f1_score']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, IoU: {metrics['pixel_iou']:.4f}, DICE: {metrics['pixel_dice']:.4f}")
    
    # Calculate overall F1 (aggregated across all objects)
    if total_tp == 0 and total_fp == 0 and total_fn == 0:
        overall_f1 = 1.0
        overall_precision = 1.0
        overall_recall = 1.0
    elif total_tp == 0:
        overall_f1 = 0.0
        overall_precision = 0.0
        overall_recall = 0.0
    else:
        overall_precision = total_tp / (total_tp + total_fp)
        overall_recall = total_tp / (total_tp + total_fn)
        overall_f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn)
    
    # Calculate mean metrics (average across files)
    mean_f1 = np.mean([r['f1_score'] for r in per_file_results])
    mean_pixel_iou = np.mean([r['pixel_iou'] for r in per_file_results])
    mean_pixel_dice = np.mean([r['pixel_dice'] for r in per_file_results])
    
    # Calculate overall pixel-level metrics (aggregated across all pixels)
    overall_pixel_iou = total_pixel_intersection / total_pixel_union if total_pixel_union > 0 else 0.0
    overall_pixel_dice = 2 * total_pixel_intersection / (total_pixel_gt + total_pixel_pred) if (total_pixel_gt + total_pixel_pred) > 0 else 0.0
    
    # Summary results
    results = {
        'overall_f1': overall_f1,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_pixel_iou': overall_pixel_iou,
        'overall_pixel_dice': overall_pixel_dice,
        'mean_f1': mean_f1,
        'mean_pixel_iou': mean_pixel_iou,
        'mean_pixel_dice': mean_pixel_dice,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'n_pairs': len(pairs),
        'per_file_results': per_file_results
    }
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Number of pairs processed: {len(pairs)}")
    print(f"\n--- Object-Level Metrics ---")
    print(f"Overall F1 Score (aggregated): {overall_f1:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Mean F1 Score (per-file average): {mean_f1:.4f}")
    print(f"\n--- Pixel-Level Metrics ---")
    print(f"Overall Pixel IoU (aggregated): {overall_pixel_iou:.4f}")
    print(f"Overall Pixel DICE (aggregated): {overall_pixel_dice:.4f}")
    print(f"Mean Pixel IoU (per-file average): {mean_pixel_iou:.4f}")
    print(f"Mean Pixel DICE (per-file average): {mean_pixel_dice:.4f}")
    print(f"\n--- Object Counts ---")
    print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Total GT objects: {total_tp + total_fn}")
    print(f"Total Pred objects: {total_tp + total_fp}")
    
    # Save results if requested
    if save_results is not None:
        save_results = Path(save_results)
        print(f"\nSaving results to: {save_results}")
        
        with open(save_results, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Segmentation Comparison Results\n")
            f.write("="*80 + "\n\n")
            f.write(f"Ground truth directory: {dir_gt}\n")
            f.write(f"Prediction directory: {dir_pred}\n")
            f.write(f"IoU threshold: {iou_threshold}\n")
            if max_depth is not None:
                f.write(f"Depth filter: cells with centroid depth < {max_depth} px "
                        f"(axis {depth_axis}) only\n")
            f.write(f"Number of pairs: {len(pairs)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("OVERALL METRICS\n")
            f.write("="*80 + "\n")
            f.write("\n--- Object-Level Metrics ---\n")
            f.write(f"Overall F1 Score (aggregated): {overall_f1:.4f}\n")
            f.write(f"Overall Precision: {overall_precision:.4f}\n")
            f.write(f"Overall Recall: {overall_recall:.4f}\n")
            f.write(f"Mean F1 Score (per-file): {mean_f1:.4f}\n")
            f.write("\n--- Pixel-Level Metrics ---\n")
            f.write(f"Overall Pixel IoU (aggregated): {overall_pixel_iou:.4f}\n")
            f.write(f"Overall Pixel DICE (aggregated): {overall_pixel_dice:.4f}\n")
            f.write(f"Mean Pixel IoU (per-file): {mean_pixel_iou:.4f}\n")
            f.write(f"Mean Pixel DICE (per-file): {mean_pixel_dice:.4f}\n")
            f.write("\n--- Object Counts ---\n")
            f.write(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}\n\n")
            
            f.write("="*80 + "\n")
            f.write("PER-FILE RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for result in per_file_results:
                f.write(f"GT: {Path(result['gt_file']).name}\n")
                f.write(f"Pred: {Path(result['pred_file']).name}\n")
                f.write(f"  F1: {result['f1_score']:.4f}, Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}\n")
                f.write(f"  Pixel IoU: {result['pixel_iou']:.4f}, Pixel DICE: {result['pixel_dice']:.4f}\n")
                f.write(f"  TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}\n")
                f.write(f"  GT objects: {result['n_gt_objects']}, Pred objects: {result['n_pred_objects']}\n\n")
    
    return results


if __name__ == "__main__":
    """
    Example usage:
    
    # Compare single pair
    f1 = compare_segmentation_pair(
        "path/to/file_masks.tif",
        "path/to/file_restored_masks.tif",
        iou_threshold=0.5,
        return_detailed=True
    )
    
    # Batch compare with position matching (sort and match by index)
    results = batch_compare_segmentations(
        dir_gt="path/to/ground_truth_dir",
        dir_pred="path/to/prediction_dir",
        pattern_gt="_masks.tif",
        pattern_pred="_restored_masks.tif",
        iou_threshold=0.5,
        save_results="comparison_results.txt",
        match_by="position"  # Sorts both dirs and matches by index
    )
    
    # Batch compare with similarity matching (BEST for mixed crops or different naming)
    results = batch_compare_segmentations(
        dir_gt="path/to/ground_truth_dir",
        dir_pred="path/to/prediction_dir",
        pattern_gt="*_masks.tif",
        pattern_pred="*.tif",
        iou_threshold=0.5,
        match_by="similarity"  # Matches by finding most similar filename
    )
    
    # Batch compare with basename matching (if files have same base name)
    results = batch_compare_segmentations(
        dir_gt="path/to/ground_truth_dir",
        dir_pred="path/to/prediction_dir",
        pattern_gt="_masks.tif",
        pattern_pred="_restored_masks.tif",
        iou_threshold=0.5,
        match_by="basename"  # Matches file1_masks.tif with file1_restored_masks.tif
    )
    """
    pass
