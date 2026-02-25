#!/usr/bin/env python3
"""
Command-line interface for comparing volumetric segmentations.

Usage examples:
    # Compare two single files
    python compare_segmentations_cli.py single file1_masks.tif file1_restored_masks.tif
    
    # Compare directories with default patterns
    python compare_segmentations_cli.py batch /path/to/gt_dir /path/to/pred_dir
    
    # Compare directories with custom patterns
    python compare_segmentations_cli.py batch /path/to/gt_dir /path/to/pred_dir \
        --pattern_gt _seg.tif --pattern_pred _pred.tif --iou 0.6 --output results.txt
"""

import argparse
import sys
from pathlib import Path

# Add script directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from compare_segmentations import compare_segmentation_pair, batch_compare_segmentations


def main():
    parser = argparse.ArgumentParser(
        description='Compare volumetric tif segmentations using object-level F1 score',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Comparison mode')
    subparsers.required = True
    
    # Single file comparison
    single_parser = subparsers.add_parser('single', help='Compare two single files')
    single_parser.add_argument('gt_file', type=str, help='Ground truth segmentation file')
    single_parser.add_argument('pred_file', type=str, help='Predicted segmentation file')
    single_parser.add_argument('--iou', type=float, default=0.5,
                               help='IoU threshold for matching (default: 0.5)')
    single_parser.add_argument('--detailed', action='store_true',
                               help='Print detailed metrics')
    
    # Batch directory comparison
    batch_parser = subparsers.add_parser('batch', help='Compare matching files in two directories')
    batch_parser.add_argument('gt_dir', type=str, help='Directory with ground truth segmentations')
    batch_parser.add_argument('pred_dir', type=str, help='Directory with predicted segmentations')
    batch_parser.add_argument('--pattern_gt', type=str, default='_masks.tif',
                              help='File pattern for ground truth (default: _masks.tif)')
    batch_parser.add_argument('--pattern_pred', type=str, default='_restored_masks.tif',
                              help='File pattern for predictions (default: _restored_masks.tif)')
    batch_parser.add_argument('--iou', type=float, default=0.5,
                              help='IoU threshold for matching (default: 0.5)')
    batch_parser.add_argument('--output', '-o', type=str, default=None,
                              help='Output file for detailed results (optional)')
    
    args = parser.parse_args()
    
    # Run comparison
    try:
        if args.mode == 'single':
            print("="*80)
            print("Single File Comparison")
            print("="*80)
            
            result = compare_segmentation_pair(
                file_gt=args.gt_file,
                file_pred=args.pred_file,
                iou_threshold=args.iou,
                return_detailed=args.detailed
            )
            
            print("\n" + "="*80)
            print("Comparison Complete!")
            print("="*80)
            
        elif args.mode == 'batch':
            print("="*80)
            print("Batch Directory Comparison")
            print("="*80)
            
            results = batch_compare_segmentations(
                dir_gt=args.gt_dir,
                dir_pred=args.pred_dir,
                pattern_gt=args.pattern_gt,
                pattern_pred=args.pattern_pred,
                iou_threshold=args.iou,
                save_results=args.output
            )
            
            if results is None:
                print("\nERROR: No matching pairs found or comparison failed")
                sys.exit(1)
        
        print("\nSuccess!")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
