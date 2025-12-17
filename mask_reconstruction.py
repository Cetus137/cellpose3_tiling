from cellpose.models import CellposeModel
import tifffile as tiff
import numpy as np
import scipy.ndimage as ndi
from cellpose.dynamics import compute_masks
from segmentation import segment_zstack
import cellpose
from natsort import natsorted
import os
from pathlib import Path
import glob


def reconstruction_masks(seg_dir,
                         output_dir,
                         timepoint,
                         flow_threshold=0.4,
                         min_size=5000,
                         do_3D=True,
                         cellprob_threshold=0.0):
        """
        Reconstruct masks from blurred distance and cell probability maps.
        Parameters:
        ------------------------------------------------------------------  
        dP_blur: numpy array
            Blurred distance maps (shape: [3, z, y, x] for 3D)
        cell_prob_blur: numpy array
            Blurred cell probability maps (shape: [z, y, x] for 3D)
        flow_threshold: float
            Threshold for flow magnitude to consider a pixel as part of a cell
        min_size: int
            Minimum size of objects to keep in the final masks     
        do_3D: bool
            Whether to perform 3D segmentation (True) or 2D segmentation (False)
        cellprob_threshold: float
            Threshold for cell probability to consider a pixel as part of a cell
        Returns:
        ------------------------------------------------------------------
        masks: numpy array
            Segmentation masks (shape: [z, y, x] for 3D)
        """
        # first find all tif files in the directory
        tif_files = [f for f in os.listdir(seg_dir) if f.endswith('.tif')]
        tif_files = natsorted(tif_files)
        # find the specific timepoint files
        flow_files     = [f for f in tif_files if f"restored_timepoint_{timepoint:04d}" in f and "flows"    in f]
        cellprob_files = [f for f in tif_files if f"restored_timepoint_{timepoint:04d}" in f and "cellprob" in f]  

        flow_xy_path     = [f for f in flow_files     if "_xy_" in f][0]  
        flow_xz_path     = [f for f in flow_files     if "_xz_" in f][0]
        flow_yz_path     = [f for f in flow_files     if "_yz_" in f][0]
        cellprob_xy_path = [f for f in cellprob_files if "_xy_" in f][0]
        cellprob_xz_path = [f for f in cellprob_files if "_xz_" in f][0]
        cellprob_yz_path = [f for f in cellprob_files if "_yz_" in f][0]


        # Load flow and cell probability maps
        flow_xy     = tiff.imread(os.path.join(seg_dir, flow_xy_path))
        flow_xz     = tiff.imread(os.path.join(seg_dir, flow_xz_path))
        flow_yz     = tiff.imread(os.path.join(seg_dir, flow_yz_path))
        cellprob_xy = tiff.imread(os.path.join(seg_dir, cellprob_xy_path))
        cellprob_xz = tiff.imread(os.path.join(seg_dir, cellprob_xz_path))
        cellprob_yz = tiff.imread(os.path.join(seg_dir, cellprob_yz_path))

        print('Shapes before reconstruction:')
        print(f"flow_yz: {flow_yz.shape}, cellprob_yz: {cellprob_yz.shape}")
        print(f"flow_xz: {flow_xz.shape}, cellprob_xz: {cellprob_xz.shape}")
        print(f"flow_xy: {flow_xy.shape}, cellprob_xy: {cellprob_xy.shape}")


        #first tranpose the xz and yz views to match the original orientation
        flow_yz     = np.transpose(flow_yz,     (0,2, 3, 1))
        cellprob_yz = np.transpose(cellprob_yz, (1, 2, 0))
        flow_xz     = np.transpose(flow_xz,     (0,2, 1, 3))
        cellprob_xz = np.transpose(cellprob_xz, (1, 0, 2))

        print('Shapes after transpose:')
        print(f"flow_yz: {flow_yz.shape}, cellprob_yz: {cellprob_yz.shape}")
        print(f"flow_xz: {flow_xz.shape}, cellprob_xz: {cellprob_xz.shape}")
        print(f"flow_xy: {flow_xy.shape}, cellprob_xy: {cellprob_xy.shape}")    

        # Average flows and cell probabilities from different views
        flowsx = (flow_xy[0] + flow_xz[0])
        flowsy = (flow_xy[1] + flow_yz[0])
        flowsz = (flow_xz[1] + flow_yz[1])
        cellprob = (cellprob_xy + cellprob_xz + cellprob_yz)
        dP = np.array([flowsz, flowsy, flowsx])

        masks = compute_masks(dP, 
                             cellprob, 
                             flow_threshold=flow_threshold, 
                             min_size=min_size, 
                             do_3D=do_3D, 
                             cellprob_threshold=cellprob_threshold)
        
        # Save masks
        output_path = os.path.join(output_dir, f'restored_timepoint_{timepoint:04d}_segmented.tif')
        print(f"Saving masks to {output_path}")
        tiff.imwrite(output_path, masks.astype(np.uint16))
        return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct segmentation masks from flow and cell probability maps.")

    parser.add_argument("--seg_dir"           , type=str  , required=True, help="Directory containing flow and cell probability maps.")
    parser.add_argument("--output_dir"        , type=str  , required=True, help="Directory to save the reconstructed masks.")
    parser.add_argument("--timepoint"         , type=int  , required=True, help="Timepoint index to process.")
    parser.add_argument("--flow_threshold"    , type=float, default=0.4,   help="Flow magnitude threshold.")
    parser.add_argument("--min_size"          , type=int  , default=5000,  help="Minimum size of objects to keep.")
    parser.add_argument("--do_3D"             , action='store_true',       help="Whether to perform 3D segmentation.")
    parser.add_argument("--cellprob_threshold", type=float , default=0.0,  help="Cell probability threshold.")

    args = parser.parse_args()

    reconstruction_masks(
        seg_dir=args.seg_dir,
        output_dir=args.output_dir,
        timepoint=args.timepoint,
        flow_threshold=args.flow_threshold,
        min_size=args.min_size,
        do_3D=args.do_3D,
        cellprob_threshold=args.cellprob_threshold
    )
        


