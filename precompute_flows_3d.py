"""
Precompute omnipose 3D flow fields for training tiles.

For each non-empty mask tile, computes the full-tile lbl tensor (omnipose
heat-diffusion flows + boundary + distance) and saves it as
{tile}_flows3d.npy alongside the mask. During training, FileTrainSet loads
these directly, eliminating per-batch flow computation.

Supports SLURM array jobs via --task_id / --n_tasks.
"""
import glob, os, argparse
import numpy as np
import torch
import tifffile as tiff

from cellpose_omni import io
from omnipose.core import masks_to_flows_batch, batch_labels
import omnipose.utils

io.logger_setup()


def precompute_flows_3d(train_dir, use_gpu=True, task_id=0, n_tasks=1):
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_mask_files = sorted(glob.glob(os.path.join(train_dir, '*_masks.tif')))
    mask_files = all_mask_files[task_id::n_tasks]
    print(f"Task {task_id}/{n_tasks}: {len(mask_files)} of {len(all_mask_files)} tiles")

    skipped_exists = 0
    skipped_empty  = 0
    computed       = 0

    for i, mask_path in enumerate(mask_files):
        img_path   = mask_path.replace('_masks.tif', '.tif')
        flows_path = img_path.replace('.tif', '_flows3d.npy')

        if os.path.exists(flows_path):
            skipped_exists += 1
            continue

        mask = np.squeeze(tiff.imread(mask_path))
        if mask.max() == 0:
            skipped_empty += 1
            continue

        print(f"[{i+1}/{len(mask_files)}] {os.path.basename(mask_path)}")

        mask = omnipose.utils.format_labels(mask)
        tyx  = mask.shape  # full tile shape, e.g. (256, 288, 288)

        labels = mask[np.newaxis].astype(np.float32)  # (1,Z,Y,X) numpy array
        links  = [None]

        out = masks_to_flows_batch(
            labels, links,
            device=device,
            omni=True, dim=3,
            affinity_field=False,
        )[:-2]

        X      = out[:-1]
        slices = out[-1]
        masks_out, bd, T, mu = [
            torch.stack([x[(Ellipsis,) + slc] for slc in slices]) for x in X
        ]
        lbl = batch_labels(
            masks_out, bd, T, mu, tyx,
            dim=3, nclasses=3,
            device=torch.device('cpu'),
        )

        np.save(flows_path, lbl[0].numpy().astype(np.float16))
        computed += 1
        print(f"  Saved: {os.path.basename(flows_path)}")

    print(f"\nDone — computed: {computed}, "
          f"skipped (exists): {skipped_exists}, "
          f"skipped (empty): {skipped_empty}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Precompute omnipose 3D flow fields for training tiles')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory containing *_masks.tif tiles')
    parser.add_argument('--no_gpu', action='store_true',
                        help='Disable GPU (default: use GPU)')
    parser.add_argument('--task_id', type=int, default=0,
                        help='Array task index (0-based). Set via $SLURM_ARRAY_TASK_ID.')
    parser.add_argument('--n_tasks', type=int, default=1,
                        help='Total number of array tasks.')
    args = parser.parse_args()

    precompute_flows_3d(
        args.train_dir,
        use_gpu=not args.no_gpu,
        task_id=args.task_id,
        n_tasks=args.n_tasks,
    )
