import tifffile as tiff
import numpy as np
import glob, os, argparse
from pathlib import Path


def _save_slices(img, mask, axis, view_tag, basename, out_dir, counters):
    """Iterate over `axis` of img/mask, save non-empty 2D slices."""
    for i in range(mask.shape[axis]):
        sl = tuple(i if d == axis else slice(None) for d in range(mask.ndim))
        mask_slice = mask[sl]
        if mask_slice.max() == 0:
            counters['skipped_empty_slice'] += 1
            continue

        out_img  = os.path.join(out_dir, f"{basename}_{view_tag}{i:04d}.tif")
        out_mask = os.path.join(out_dir, f"{basename}_{view_tag}{i:04d}_masks.tif")

        if os.path.exists(out_img) and os.path.exists(out_mask):
            counters['skipped_exists'] += 1
            continue

        tiff.imwrite(out_img,  img[sl])
        tiff.imwrite(out_mask, mask_slice)
        counters['saved'] += 1


def prepare_2d(train_dir, task_id=0, n_tasks=1):
    out_dir = os.path.join(train_dir, 'training_2D')
    os.makedirs(out_dir, exist_ok=True)

    all_mask_files = sorted(glob.glob(os.path.join(train_dir, '*_masks.tif')))
    mask_files = all_mask_files[task_id::n_tasks]
    print(f"Task {task_id}/{n_tasks}: {len(mask_files)} of {len(all_mask_files)} tiles")

    counters = dict(saved=0, skipped_exists=0, skipped_empty_tile=0, skipped_empty_slice=0)

    for mask_path in mask_files:
        img_path = mask_path.replace('_masks.tif', '.tif')
        if not os.path.exists(img_path):
            print(f"  WARNING: no raw file for {os.path.basename(mask_path)}, skipping")
            continue

        mask = np.squeeze(tiff.imread(mask_path))
        if mask.max() == 0:
            counters['skipped_empty_tile'] += 1
            continue

        img      = np.squeeze(tiff.imread(img_path))
        basename = Path(img_path).stem

        # XY slices (axis 0 = Z)
        _save_slices(img, mask, axis=0, view_tag='xy_s', basename=basename,
                     out_dir=out_dir, counters=counters)
        # XZ slices (axis 1 = Y)
        _save_slices(img, mask, axis=1, view_tag='xz_s', basename=basename,
                     out_dir=out_dir, counters=counters)
        # YZ slices (axis 2 = X)
        _save_slices(img, mask, axis=2, view_tag='yz_s', basename=basename,
                     out_dir=out_dir, counters=counters)

    print(f"\nDone — saved: {counters['saved']}, "
          f"skipped (exists): {counters['skipped_exists']}, "
          f"skipped (empty tile): {counters['skipped_empty_tile']}, "
          f"skipped (empty slice): {counters['skipped_empty_slice']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract non-empty XY/XZ/YZ slices from 3D tiles for 2D cellpose training")
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory containing 3D *_masks.tif tiles; output written to train_dir/training_2D/')
    parser.add_argument('--task_id', type=int, default=0,
                        help='Array task index (0-based). Set via $SLURM_ARRAY_TASK_ID.')
    parser.add_argument('--n_tasks', type=int, default=1,
                        help='Total number of array tasks.')
    args = parser.parse_args()

    prepare_2d(args.train_dir, task_id=args.task_id, n_tasks=args.n_tasks)
