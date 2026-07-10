"""
cpSAM (cellpose 4) training on 2D slices.

Uses the same data selection logic as cellpose_training.py but initialises a
CellposeModel with the default cpsam pretrained backbone and uses the cellpose 4
train_seg API (AdamW, channel_axis instead of channels, bfloat16 weights).

Reuses precomputed _flows.tif files produced by precompute_flows.py.
"""
from cellpose import io, models, train
import glob, os, re, json, argparse, random
io.logger_setup()

# Shared utilities — same selection/filtering logic as cellpose_training.py
from cellpose_training import _diverse_sample, _filter_sparse


def _find_latest_checkpoint(models_dir, model_name):
    """
    Return (path, relative_epoch) of the highest-numbered epoch checkpoint,
    or (None, 0) if none exist.
    """
    pattern = os.path.join(models_dir, f'{model_name}_epoch_*')
    epoch_re = re.compile(rf'^{re.escape(model_name)}_epoch_(\d+)$')
    best_path, best_epoch = None, 0
    for c in glob.glob(pattern):
        m = epoch_re.match(os.path.basename(c))
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_path = c
    return best_path, best_epoch


def _meta_path(models_dir, model_name):
    return os.path.join(models_dir, f'{model_name}.meta.json')


def _read_meta(models_dir, model_name):
    p = _meta_path(models_dir, model_name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"start_epoch_offset": 0}


def _write_meta(models_dir, model_name, start_epoch_offset):
    os.makedirs(models_dir, exist_ok=True)
    with open(_meta_path(models_dir, model_name), 'w') as f:
        json.dump({"start_epoch_offset": start_epoch_offset}, f)


def _done_path(models_dir, model_name):
    return os.path.join(models_dir, f'{model_name}.done')


def train_model(train_dir, max_tiles=10000, seed=42, min_label_pixels=50,
                n_epochs=500, batch_size=4, learning_rate=5e-5, model_name='cpsam_ph3',
                save_path=None, resume=False, resume_from=None):

    if save_path is None:
        save_path = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(save_path, 'models')

    # --- resume logic ---
    start_epoch = 0
    pretrained_model = "cpsam"

    if resume_from:
        # Load a specific model file; treat as epoch 0 (full LR schedule from start).
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"--resume_from path not found: {resume_from}")
        print(f"Warm-starting from {resume_from}")
        pretrained_model = resume_from

    elif resume:
        if os.path.exists(_done_path(models_dir, model_name)):
            print(f"Training already complete ({model_name}). "
                  "Delete the .done file to force re-training.")
            return

        ckpt_path, rel_epoch = _find_latest_checkpoint(models_dir, model_name)
        if ckpt_path is None:
            print("No checkpoint found — starting from scratch.")
        else:
            meta = _read_meta(models_dir, model_name)
            start_epoch = meta["start_epoch_offset"] + rel_epoch
            print(f"Resuming from {os.path.basename(ckpt_path)} "
                  f"(absolute epoch {start_epoch}/{n_epochs})")
            pretrained_model = ckpt_path

    remaining_epochs = n_epochs - start_epoch
    if remaining_epochs <= 0:
        print(f"Nothing to do: start_epoch={start_epoch} >= n_epochs={n_epochs}")
        return

    # --- data selection (unchanged) ---
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

    model = models.CellposeModel(gpu=True, pretrained_model=pretrained_model)

    # Record start_epoch before training so _find_latest_checkpoint results can
    # be converted to absolute epochs on the next resume.
    _write_meta(models_dir, model_name, start_epoch)

    train.train_seg(
        model.net,
        train_files=train_files,
        train_labels_files=train_labels_files,
        test_files=test_files,
        test_labels_files=test_labels_files,
        channel_axis=None,       # single-channel grayscale
        compute_flows=False,     # reuse precomputed _flows.tif
        load_files=True,         # cellpose4 bug: load_files=False makes the training loop
                                 # pass channel_axis to _get_batch which doesn't accept it;
                                 # 10k 2D tiles (~1.3 GB) fits in RAM without issue
        min_train_masks=0,       # already filtered by _filter_sparse
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=remaining_epochs,
        weight_decay=0.1,
        save_every=50,
        save_each=True,          # write cpsam_ph3_epoch_NNNN checkpoints for resume
        save_path=save_path,
        model_name=model_name,
    )

    # Mark training as complete so accidental re-runs with --resume are no-ops.
    open(_done_path(models_dir, model_name), 'w').close()
    print(f"Training complete. Model saved to {os.path.join(models_dir, model_name)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train cpSAM (cellpose 4) on 2D slices')
    parser.add_argument('--train_dir', type=str,
                        default='/users/kir-fritzsche/aif490/devel/tissue_analysis/'
                                'segmentation_scripts/for_training/ph3/training_raw/training_2D')
    parser.add_argument('--save_path', type=str,
                        default='/users/kir-fritzsche/aif490/devel/tissue_analysis/'
                                'segmentation_scripts',
                        help='Root dir for models/ subdir (default: script directory)')
    parser.add_argument('--max_tiles',        type=int,   default=10000)
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--min_label_pixels', type=int,   default=50)
    parser.add_argument('--n_epochs',         type=int,   default=500)
    parser.add_argument('--batch_size',       type=int,   default=4)
    parser.add_argument('--learning_rate',    type=float, default=5e-5)
    parser.add_argument('--model_name',       type=str,   default='cpsam_ph3')
    parser.add_argument('--resume',           action='store_true',
                        help='Resume from the latest epoch checkpoint if one exists')
    parser.add_argument('--resume_from',      type=str,   default=None,
                        help='Warm-start from a specific model file path (overrides --resume)')
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
        save_path=args.save_path,
        resume=args.resume,
        resume_from=args.resume_from,
    )
