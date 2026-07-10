"""
File-based 3D omnipose training with 2-channel seeded input.

Input data layout (per sample):
  <sample>_composite.tif        — shape (2, Z, Y, X), dtype uint16/float
                                   ch0 = raw intensity
                                   ch1 = nuclear seed masks (instance labels)
  <sample>_composite_masks.tif  — shape (Z, Y, X), whole-cell segmentation masks
  <sample>_composite_flows3d.npy — precomputed omnipose label tensor
                                   (generate with precompute_flows_3d.py
                                    pointed at the same train_dir)

Channel normalisation:
  ch0 (raw)    — omnipose.utils.normalize99
  ch1 (seeds)  — binarised to {0, 1}  (presence/absence of nuclear signal)
"""
import glob, os, datetime, argparse, random
import numpy as np
import torch
import tifffile as tiff

from cellpose_omni import models, io
from omnipose.data import train_set
from omnipose.core import random_crop_warp, masks_to_flows_batch, batch_labels
import omnipose.utils

io.logger_setup()

NCHAN = 2


def _load_composite(path):
    """Load a (2, Z, Y, X) composite tif and return a normalised (2, Z, Y, X) float32 array."""
    raw = tiff.imread(path).astype(np.float32)

    if raw.ndim == 4 and raw.shape[0] == NCHAN:
        img = np.empty_like(raw)
        img[0] = omnipose.utils.normalize99(raw[0])
        img[1] = (raw[1] > 0).astype(np.float32)   # binarise nuclear seeds
    elif raw.ndim == 3:
        # single-channel file accidentally passed — promote and leave ch1 as zeros
        img = np.zeros((NCHAN,) + raw.shape, dtype=np.float32)
        img[0] = omnipose.utils.normalize99(raw)
    else:
        raise ValueError(f"Unexpected composite shape {raw.shape} in {path}")

    return img


class SeededTrainSet(train_set):
    """
    FileTrainSet for 2-channel seeded omnipose training.

    Expects composites (*_composite.tif) and paired masks (*_composite_masks.tif).
    If precomputed *_composite_flows3d.npy files exist, uses the fast path.
    """

    def __init__(self, img_files, mask_files, flows_files=None, **kwargs):
        self.img_files   = img_files
        self.mask_files  = mask_files
        self.flows_files = flows_files
        n = len(img_files)
        super().__init__([None] * n, [None] * n, links=[None] * n, **kwargs)

    def __getitem__(self, inds):
        if isinstance(inds, int):
            inds = [inds]
        if self.flows_files is not None:
            return self._getitem_precomputed(inds)
        return self._getitem_live(inds)

    def _getitem_precomputed(self, inds):
        nimg = len(inds)
        imgi = np.zeros((nimg, NCHAN) + self.tyx, np.float32)
        lbl_list = []

        for i, idx in enumerate(inds):
            img      = _load_composite(self.img_files[idx])           # (2, Z, Y, X)
            lbl_full = np.load(self.flows_files[idx]).astype(np.float32)

            starts = [
                np.random.randint(0, max(1, img.shape[d + 1] - self.tyx[d]))
                for d in range(self.dim)
            ]
            spatial = tuple(slice(st, st + c) for st, c in zip(starts, self.tyx))
            imgi[i] = img[(slice(None),) + spatial]
            lbl_list.append(torch.tensor(lbl_full[(slice(None),) + spatial], dtype=torch.float32))

        return torch.tensor(imgi), torch.stack(lbl_list), inds

    def _getitem_live(self, inds):
        nimg   = len(inds)
        imgi   = np.zeros((nimg, NCHAN) + self.tyx, np.float32)
        labels = np.zeros((nimg,) + self.tyx, np.float32)
        links  = [None] * nimg

        for i, idx in enumerate(inds):
            img  = _load_composite(self.img_files[idx])               # (2, Z, Y, X)
            mask = np.squeeze(tiff.imread(self.mask_files[idx]))
            mask = omnipose.utils.format_labels(mask)

            imgi[i], labels[i], _ = random_crop_warp(
                img=img, Y=mask,
                tyx=self.tyx, v1=self.v1, v2=self.v2,
                nchan=NCHAN, rescale=self.rescale[idx],
                scale_range=self.scale_range,
                gamma_range=self.gamma_range,
                do_flip=self.do_flip, ind=idx,
            )

        out = masks_to_flows_batch(
            labels, links,
            device=self.device,
            omni=True, dim=self.dim,
            affinity_field=self.affinity_field,
        )[:-2]

        X = out[:-1]
        slices = out[-1]
        masks, bd, T, mu = [
            torch.stack([x[(Ellipsis,) + slc] for slc in slices]) for x in X
        ]
        lbl = batch_labels(
            masks, bd, T, mu, self.tyx,
            dim=self.dim, nclasses=3,
            device=torch.device('cpu'),
        )
        return torch.tensor(imgi), lbl, inds


def _build_lr_schedule(learning_rate, n_epochs):
    LR = np.linspace(0, learning_rate, 10)
    if n_epochs > 250:
        LR = np.append(LR, learning_rate * np.ones(n_epochs - 100))
        for _ in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    else:
        LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
    return LR


def _net_state_dict(model):
    sd = model.net.state_dict()
    if any(k.startswith('module.') for k in sd):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    return sd


def train_omnipose_seeded(
    train_dir,
    model_name='omnipose_3d_seeded',
    n_epochs=4000,
    batch_size=1,
    learning_rate=0.005,
    save_every=10,
    weight_decay=1e-5,
    momentum=0.9,
    num_workers=4,
):
    save_dir = os.path.join(train_dir, 'models')
    os.makedirs(save_dir, exist_ok=True)

    all_mask_files = sorted(glob.glob(os.path.join(train_dir, '*_composite_masks.tif')))
    img_files, mask_files, flows_files = [], [], []
    for mf in all_mask_files:
        mask = tiff.imread(mf)
        if np.count_nonzero(mask) / mask.size < 0.01:
            continue
        imf = mf.replace('_composite_masks.tif', '_composite.tif')
        if not os.path.exists(imf):
            continue
        img_files.append(imf)
        mask_files.append(mf)
        flows_files.append(imf.replace('_composite.tif', '_composite_flows3d.npy'))

    if not img_files:
        raise RuntimeError(
            f"No paired *_composite.tif / *_composite_masks.tif found in {train_dir}"
        )

    ready = [os.path.exists(f) for f in flows_files]
    n_precomputed = sum(ready)
    if n_precomputed == 0:
        print("No precomputed flows found — using on-the-fly computation (slow).")
        flows_files_arg = None
    else:
        if n_precomputed < len(img_files):
            img_files   = [f for f, r in zip(img_files,   ready) if r]
            mask_files  = [f for f, r in zip(mask_files,  ready) if r]
            flows_files = [f for f, r in zip(flows_files, ready) if r]
            print(f"Training on {n_precomputed}/{len(ready)} tiles with precomputed flows")
        else:
            print(f"Training on {len(img_files)} tiles with precomputed flows (fast path)")
        flows_files_arg = flows_files

    rng = random.Random(42)
    indices = list(range(len(img_files)))
    rng.shuffle(indices)
    n_test = max(1, int(0.1 * len(indices)))
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    def _split(lst): return [lst[i] for i in test_idx], [lst[i] for i in train_idx]

    test_img,   img_files   = _split(img_files)
    test_mask,  mask_files  = _split(mask_files)
    test_flows, flows_files = _split(flows_files) if flows_files_arg else (None, None)
    print(f"  train: {len(img_files)}  test: {len(test_img)}")

    model = models.CellposeModel(
        gpu=True, pretrained_model=False,
        nchan=NCHAN, nclasses=3, dim=3, omni=True,
    )

    tyx = (128,) * 3
    dataset_kwargs = dict(
        rescale=False, diam_train=None, tyx=tyx,
        scale_range=1.0, omni=True, dim=3, nchan=NCHAN, nclasses=3,
        device=model.device, affinity_field=False,
    )

    dataset = SeededTrainSet(
        img_files, mask_files, flows_files=flows_files, **dataset_kwargs
    )
    test_dataset = SeededTrainSet(
        test_img, test_mask, flows_files=test_flows, **dataset_kwargs
    )

    def _make_loader(ds, shuffle):
        sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.sampler.RandomSampler(ds) if shuffle
            else torch.utils.data.sampler.SequentialSampler(ds),
            batch_size=batch_size, drop_last=False,
        )
        return torch.utils.data.DataLoader(
            ds, batch_size=1, shuffle=False, sampler=sampler,
            collate_fn=ds.collate_fn, worker_init_fn=ds.worker_init_fn,
            num_workers=num_workers, pin_memory=True,
        )

    loader      = _make_loader(dataset,      shuffle=True)
    test_loader = _make_loader(test_dataset, shuffle=False)

    LR = _build_lr_schedule(learning_rate, n_epochs)
    model.net.mkldnn = False
    model.autocast   = False
    model._set_optimizer(LR[0], momentum, weight_decay, SGD=True)
    model._set_criterion()

    print(f"Starting: {n_epochs} epochs, batch_size={batch_size}, "
          f"lr={learning_rate}, crop={tyx}, nchan={NCHAN}, device={model.device}")

    for iepoch in range(n_epochs):
        model._set_learning_rate(LR[iepoch])
        lsum, nsum = 0, 0

        for _, (batch_imgs, batch_lbl, _) in enumerate(loader):
            batch_imgs = batch_imgs.to(model.device)
            batch_lbl  = batch_lbl.to(model.device)
            loss = model._train_step(batch_imgs, batch_lbl)
            lsum += loss
            nsum += len(batch_imgs)

        print(f"Epoch {iepoch:4d}/{n_epochs}  "
              f"loss={lsum / max(nsum, 1):.5f}  "
              f"[{datetime.datetime.now():%H:%M:%S}]")

        if iepoch == 5 or (iepoch + 1) % save_every == 0:
            tlsum, tnsum = 0, 0
            for _, (test_imgs, test_lbl, _) in enumerate(test_loader):
                test_imgs = test_imgs.to(model.device)
                test_lbl  = test_lbl.to(model.device)
                tlsum += model._test_eval(test_imgs, test_lbl)
                tnsum += len(test_imgs)
            print(f"  test_loss={tlsum / max(tnsum, 1):.5f}")

        if (iepoch + 1) % save_every == 0:
            ckpt = os.path.join(save_dir, f'{model_name}_ep{iepoch + 1:04d}')
            torch.save(_net_state_dict(model), ckpt)
            print(f"  Saved: {os.path.basename(ckpt)}")

    final = os.path.join(save_dir, model_name)
    torch.save(_net_state_dict(model), final)
    print(f"\nDone. Final model: {final}")
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='3D omnipose training with 2-channel nuclear-seeded input')
    parser.add_argument('--train_dir',     type=str, required=True,
                        help='Directory containing *_composite.tif and *_composite_masks.tif')
    parser.add_argument('--model_name',    type=str,   default='omnipose_3d_seeded')
    parser.add_argument('--n_epochs',      type=int,   default=4000)
    parser.add_argument('--batch_size',    type=int,   default=1)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--save_every',    type=int,   default=10)
    parser.add_argument('--num_workers',   type=int,   default=4)
    args = parser.parse_args()

    train_omnipose_seeded(
        train_dir=args.train_dir,
        model_name=args.model_name,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        num_workers=args.num_workers,
    )
