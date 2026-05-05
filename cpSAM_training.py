"""
cpSAM (cellpose 4) training on 2D slices.

Uses the same data selection logic as cellpose_training.py but initialises a
CellposeModel with the default cpsam pretrained backbone and uses the cellpose 4
train_seg API (AdamW, channel_axis instead of channels, bfloat16 weights).

Reuses precomputed _flows.tif files produced by precompute_flows.py.
"""
from cellpose import io, models, train
import glob, os, argparse, random
io.logger_setup()

# Shared utilities — same selection/filtering logic as cellpose_training.py
from cellpose_training import _diverse_sample, _filter_sparse


def train_model(train_dir, max_tiles=10000, seed=42, min_label_pixels=50,
                n_epochs=500, batch_size=4, learning_rate=5e-5, model_name='cpsam_ph3'):

    all_tifs  = sorted(glob.glob(os.path.join(train_dir, '*.tif')))
    raw_files = [f for f in all_tifs if '_masks' not in f and '_flows' not in f]

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

    rng = random.Random(seed)
    indices = list(range(len(train_files)))
    rng.shuffle(indices)
    n_test = max(1, int(0.1 * len(indices)))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    test_files         = [train_files[i]        for i in test_idx]
    test_labels_files  = [train_labels_files[i] for i in test_idx]
    train_files        = [train_files[i]        for i in train_idx]
    train_labels_files = [train_labels_files[i] for i in train_idx]
    print(f"  train: {len(train_files)}  test: {len(test_files)}")

    for f, lf in zip(train_files[:3], train_labels_files[:3]):
        print(f"  {os.path.basename(f)}  ->  {os.path.basename(lf)}")

    # pretrained_model="cpsam" is the default in cellpose 4
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    train.train_seg(
        model.net,
        train_files=train_files,
        train_labels_files=train_labels_files,
        test_files=test_files,
        test_labels_files=test_labels_files,
        channel_axis=None,       # single-channel grayscale
        compute_flows=False,     # reuse precomputed _flows.tif
        load_files=False,
        min_train_masks=0,       # already filtered by _filter_sparse; avoids cellpose4 bug with load_files=False
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        weight_decay=0.1,
        save_every=50,
        model_name=model_name,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train cpSAM (cellpose 4) on 2D slices')
    parser.add_argument('--train_dir', type=str,
                        default='/users/kir-fritzsche/aif490/devel/tissue_analysis/'
                                'segmentation_scripts/for_training/ph3/training_raw/training_2D')
    parser.add_argument('--max_tiles',        type=int,   default=10000)
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--min_label_pixels', type=int,   default=50)
    parser.add_argument('--n_epochs',         type=int,   default=500)
    parser.add_argument('--batch_size',       type=int,   default=4)
    parser.add_argument('--learning_rate',    type=float, default=5e-5)
    parser.add_argument('--model_name',       type=str,   default='cpsam_ph3')
    args = parser.parse_args()

    train_model(
        train_dir=args.train_dir,
        max_tiles=args.max_tiles,
        seed=args.seed,
        min_label_pixels=args.min_label_pixels,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_name=args.model_name,
    )
