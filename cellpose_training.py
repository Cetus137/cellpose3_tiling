from cellpose import io, models, train
import glob, os, re, random, argparse
from collections import defaultdict
import numpy as np
import tifffile as tiff
io.logger_setup()


def _diverse_sample(train_files, max_tiles, seed):
    """
    Select up to max_tiles files maximising diversity.

    Groups files by (source 3D tile, view: xy/xz/yz), then takes evenly-spaced
    slices within each group so adjacent slices — which are nearly identical — are
    not over-represented. Budget is distributed evenly across groups; any remainder
    is filled with a random draw from what was left out.
    """
    rng = random.Random(seed)

    groups = defaultdict(list)
    for f in train_files:
        name = os.path.basename(f)
        m = re.search(r'_(xy_s|xz_s|yz_s)\d{4}\.tif$', name)
        key = (name[:m.start()], m.group(1)) if m else ('other', 'other')
        groups[key].append(f)

    for files in groups.values():
        files.sort()

    n_groups  = len(groups)
    per_group = max(1, max_tiles // n_groups)

    selected, leftover = [], []
    for files in groups.values():
        if len(files) <= per_group:
            selected.extend(files)
        else:
            indices = np.linspace(0, len(files) - 1, per_group, dtype=int)
            chosen = {files[i] for i in indices}
            selected.extend(chosen)
            leftover.extend([f for f in files if f not in chosen])

    if len(selected) < max_tiles and leftover:
        rng.shuffle(leftover)
        selected.extend(leftover[:max_tiles - len(selected)])

    if len(selected) > max_tiles:
        rng.shuffle(selected)
        selected = selected[:max_tiles]

    return selected


def _filter_sparse(train_files, train_labels_files, min_label_pixels):
    """
    Drop slices where every label instance is smaller than min_label_pixels.
    Tiny labels cause flow computation to fail after augmentation.
    """
    valid_imgs, valid_labels = [], []
    removed = 0
    for img_f, lbl_f in zip(train_files, train_labels_files):
        mask = tiff.imread(lbl_f)
        if mask.max() == 0:
            removed += 1
            continue
        counts = np.bincount(mask.ravel())
        # counts[0] is background; largest foreground label size
        if counts[1:].max() < min_label_pixels:
            removed += 1
            continue
        valid_imgs.append(img_f)
        valid_labels.append(lbl_f)
    if removed:
        print(f"Filtered {removed} slices with all labels < {min_label_pixels} px")
    return valid_imgs, valid_labels


def train_model(train_dir, max_tiles=10000, seed=42, min_label_pixels=50):
    all_tifs   = sorted(glob.glob(os.path.join(train_dir, '*.tif')))
    raw_files  = [f for f in all_tifs if '_masks' not in f and '_flows' not in f]

    train_files = [f for f in raw_files if os.path.exists(f.replace('.tif', '_flows.tif'))]
    skipped     = len(raw_files) - len(train_files)

    if len(train_files) > max_tiles:
        train_files = _diverse_sample(train_files, max_tiles, seed)
        print(f"Selected {len(train_files)} diverse tiles "
              f"({skipped} skipped — no precomputed flows)")
    else:
        print(f"Training on {len(train_files)} tiles ({skipped} skipped — no precomputed flows)")

    train_labels_files = [f.replace('.tif', '_masks.tif') for f in train_files]
    train_files, train_labels_files = _filter_sparse(
        train_files, train_labels_files, min_label_pixels)
    print(f"Final training set: {len(train_files)} tiles")

    # 10% held-out validation split (shuffled with the same seed)
    rng = random.Random(seed)
    indices = list(range(len(train_files)))
    rng.shuffle(indices)
    n_test = max(1, int(0.1 * len(indices)))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    test_files        = [train_files[i]        for i in test_idx]
    test_labels_files = [train_labels_files[i] for i in test_idx]
    train_files       = [train_files[i]        for i in train_idx]
    train_labels_files = [train_labels_files[i] for i in train_idx]
    print(f"  train: {len(train_files)}  test: {len(test_files)}")

    for f, lf in zip(train_files[:3], train_labels_files[:3]):
        print(f"  {os.path.basename(f)}  ->  {os.path.basename(lf)}")

    model = models.CellposeModel(gpu=True)

    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_files=train_files,
        train_labels_files=train_labels_files,
        test_files=test_files,
        test_labels_files=test_labels_files,
        load_files=False,
        batch_size=16,
        channels=[0, 0],
        save_every=10,
        weight_decay=0.1,
        learning_rate=1e-5,
        n_epochs=100,
        model_name="cp3_ph3",
    )
    print(f"Train losses: {train_losses}")
    print(f"Test  losses: {test_losses}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train cellpose 2D model on 2D slices')
    parser.add_argument('--train_dir', type=str,
                        default='/users/kir-fritzsche/aif490/devel/tissue_analysis/'
                                'segmentation_scripts/for_training/ph3/training_raw/training_2D')
    parser.add_argument('--max_tiles', type=int, default=10000,
                        help='Maximum tiles to select (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for tie-breaking (default: 42)')
    parser.add_argument('--min_label_pixels', type=int, default=50,
                        help='Minimum pixels per label instance; slices where all '
                             'labels are smaller are dropped (default: 50)')
    args = parser.parse_args()

    train_model(args.train_dir, max_tiles=args.max_tiles, seed=args.seed,
                min_label_pixels=args.min_label_pixels)
