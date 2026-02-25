import numpy as np
import tifffile as tiff
from pathlib import Path
from scipy import ndimage
from collections import defaultdict
import glob


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
    """
    # Get unique labels (excluding background 0)
    gt_labels = set(np.unique(seg_gt)) - {0}
    pred_labels = set(np.unique(seg_pred)) - {0}
    
    # Track best matches for each ground truth object
    matches = {}
    matched_pred = set()
    
    # For each ground truth object, find best matching prediction
    for gt_label in gt_labels:
        gt_mask = (seg_gt == gt_label)
        best_iou = 0.0
        best_pred_label = None
        
        # Find which predicted objects overlap with this ground truth object
        overlapping_pred_labels = set(np.unique(seg_pred[gt_mask])) - {0}
        
        for pred_label in overlapping_pred_labels:
            pred_mask = (seg_pred == pred_label)
            iou = calculate_iou(gt_mask, pred_mask)
            
            if iou > best_iou:
                best_iou = iou
                best_pred_label = pred_label
        
        # If best match exceeds threshold, record it
        if best_iou >= iou_threshold and best_pred_label is not None:
            matches[gt_label] = best_pred_label
            matched_pred.add(best_pred_label)
    
    # Identify unmatched objects
    unmatched_gt = gt_labels - set(matches.keys())
    unmatched_pred = pred_labels - matched_pred
    
    return matches, unmatched_gt, unmatched_pred


def calculate_object_f1(seg_gt, seg_pred, iou_threshold=0.5, return_detailed=False):
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
        
    Returns:
    --------
    f1_score : float
        Object-level F1 score between 0 and 1
    or
    metrics : dict
        Dictionary containing f1_score, precision, recall, tp, fp, fn
    """
    # Match objects between segmentations
    matches, unmatched_gt, unmatched_pred = match_objects(seg_gt, seg_pred, iou_threshold)
    
    # Calculate metrics
    tp = len(matches)  # True positives: matched ground truth objects
    fp = len(unmatched_pred)  # False positives: unmatched predicted objects
    fn = len(unmatched_gt)  # False negatives: unmatched ground truth objects
    
    # Calculate F1 score
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
    
    if return_detailed:
        return {
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'n_gt_objects': tp + fn,
            'n_pred_objects': tp + fp
        }
    
    return f1_score


def compare_segmentation_pair(file_gt, file_pred, iou_threshold=0.5, return_detailed=False):
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
    
    # Calculate F1 score
    result = calculate_object_f1(seg_gt, seg_pred, iou_threshold, return_detailed)
    
    if return_detailed:
        print(f"F1 Score: {result['f1_score']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}")
        print(f"GT objects: {result['n_gt_objects']}, Pred objects: {result['n_pred_objects']}")
    else:
        print(f"F1 Score: {result:.4f}")
    
    return result


def find_matching_pairs(dir_gt, dir_pred, pattern_gt="_masks.tif", pattern_pred="_restored_masks.tif"):
    """
    Find matching pairs of segmentation files in two directories.
    
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
        
    Returns:
    --------
    pairs : list of tuple
        List of (gt_file, pred_file) tuples
    """
    dir_gt = Path(dir_gt)
    dir_pred = Path(dir_pred)
    
    # Get all files matching the ground truth pattern
    gt_files = sorted(dir_gt.glob(f"*{pattern_gt}"))
    
    pairs = []
    
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
    
    print(f"\nFound {len(pairs)} matching pairs")
    return pairs


def batch_compare_segmentations(dir_gt, dir_pred, pattern_gt="_masks.tif", 
                                 pattern_pred="_restored_masks.tif", 
                                 iou_threshold=0.5, 
                                 save_results=None):
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
        
    Returns:
    --------
    results : dict
        Dictionary containing:
            - 'overall_f1': F1 score aggregated across all pairs
            - 'mean_f1': Mean F1 score across pairs
            - 'per_file_results': List of per-file results
            - 'total_tp', 'total_fp', 'total_fn': Aggregated counts
    """
    # Find matching pairs
    pairs = find_matching_pairs(dir_gt, dir_pred, pattern_gt, pattern_pred)
    
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
    
    for i, (gt_file, pred_file) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] Processing: {gt_file.name}")
        
        # Load and compare
        seg_gt = tiff.imread(gt_file)
        seg_pred = tiff.imread(pred_file)
        
        # Calculate metrics
        metrics = calculate_object_f1(seg_gt, seg_pred, iou_threshold, return_detailed=True)
        
        # Store results
        result_entry = {
            'gt_file': str(gt_file),
            'pred_file': str(pred_file),
            'f1_score': metrics['f1_score'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
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
        
        print(f"  F1: {metrics['f1_score']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    
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
    
    # Calculate mean F1 (average across files)
    mean_f1 = np.mean([r['f1_score'] for r in per_file_results])
    
    # Summary results
    results = {
        'overall_f1': overall_f1,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'mean_f1': mean_f1,
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
    print(f"\nOverall F1 Score (aggregated): {overall_f1:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"\nMean F1 Score (per-file average): {mean_f1:.4f}")
    print(f"\nTotal TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
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
            f.write(f"Number of pairs: {len(pairs)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("OVERALL METRICS\n")
            f.write("="*80 + "\n")
            f.write(f"Overall F1 Score (aggregated): {overall_f1:.4f}\n")
            f.write(f"Overall Precision: {overall_precision:.4f}\n")
            f.write(f"Overall Recall: {overall_recall:.4f}\n")
            f.write(f"Mean F1 Score (per-file): {mean_f1:.4f}\n")
            f.write(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}\n\n")
            
            f.write("="*80 + "\n")
            f.write("PER-FILE RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for result in per_file_results:
                f.write(f"GT: {Path(result['gt_file']).name}\n")
                f.write(f"Pred: {Path(result['pred_file']).name}\n")
                f.write(f"  F1: {result['f1_score']:.4f}, Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}\n")
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
    
    # Batch compare directories
    results = batch_compare_segmentations(
        dir_gt="path/to/ground_truth_dir",
        dir_pred="path/to/prediction_dir",
        pattern_gt="_masks.tif",
        pattern_pred="_restored_masks.tif",
        iou_threshold=0.5,
        save_results="comparison_results.txt"
    )
    """
    pass
