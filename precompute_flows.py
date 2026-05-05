from cellpose import dynamics, io
import tifffile as tiff
import numpy as np
import glob, os, argparse

io.logger_setup()


def precompute_flows(train_dir, use_gpu=True, task_id=0, n_tasks=1):
    """
    Precompute and save cellpose flow fields for every non-empty mask tile.

    Passes the image path via the `files` argument so cellpose saves flows
    automatically as {image}_flows.tif alongside each tile. On subsequent
    training runs cellpose loads these directly, skipping recalculation and
    avoiding the RAM spike that causes OOM.

    task_id / n_tasks allow SLURM array jobs to split work: each task processes
    every n_tasks-th file starting at task_id (interleaved, so load is balanced
    even if early tiles are faster/slower than later ones).

    Already-computed files are skipped, so the script is safe to re-run.
    """
    import torch
    device = torch.device('cuda') if use_gpu else torch.device('cpu')

    all_mask_files = sorted(glob.glob(os.path.join(train_dir, '*_masks.tif')))
    mask_files = all_mask_files[task_id::n_tasks]
    print(f"Task {task_id}/{n_tasks}: {len(mask_files)} of {len(all_mask_files)} tiles")

    skipped_exists = 0
    skipped_empty  = 0
    computed       = 0

    for i, mask_path in enumerate(mask_files):
        img_path   = mask_path.replace('_masks.tif', '.tif')
        flows_path = img_path.replace('.tif', '_flows.tif')

        if os.path.exists(flows_path):
            skipped_exists += 1
            continue

        mask = tiff.imread(mask_path)

        if mask.max() == 0:
            skipped_empty += 1
            continue

        print(f"[{i+1}/{len(mask_files)}] {os.path.basename(mask_path)}")
        # mask[np.newaxis] gives shape (1, Z, Y, X) — cellpose requires shape[0]==1
        # to recognise the input as a label mask and enter the computation branch.
        # Without it, a (256,256,256) mask has shape[0]=256 and is wrongly treated
        # as precomputed flows, so nothing is saved.
        dynamics.labels_to_flows([mask[np.newaxis]], files=[img_path], device=device,
                                  redo_flows=False)
        computed += 1
        print(f"  Saved: {os.path.basename(flows_path)}")

    print(f"\nDone — computed: {computed}, skipped (exists): {skipped_exists}, "
          f"skipped (empty mask): {skipped_empty}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute cellpose flows for training tiles")
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory containing *_masks.tif tiles')
    parser.add_argument('--no_gpu', action='store_true',
                        help='Disable GPU for flow computation (default: use GPU)')
    parser.add_argument('--task_id', type=int, default=0,
                        help='Array task index (0-based). Set automatically via $SLURM_ARRAY_TASK_ID.')
    parser.add_argument('--n_tasks', type=int, default=1,
                        help='Total number of array tasks.')
    args = parser.parse_args()

    precompute_flows(args.train_dir, use_gpu=not args.no_gpu,
                     task_id=args.task_id, n_tasks=args.n_tasks)
