"""
File-based 3D omnipose training.

When _flows3d.npy files are present alongside training tiles, loads precomputed
lbl tensors and takes random crops — no per-batch flow computation.
Falls back to on-the-fly flow computation if precomputed files are absent.
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


class FileTrainSet(train_set):
    """
    Subclass of omnipose.data.train_set that streams tiles from disk.

    If flows_files is provided (list of _flows3d.npy paths), uses the fast
    precomputed path: load img + lbl, random crop, return immediately.
    Otherwise falls back to on-the-fly flow computation via random_crop_warp
    + masks_to_flows_batch (slow for large 3D tiles).
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
        imgi = np.zeros((nimg, self.nchan) + self.tyx, np.float32)
        lbl_list = []

        for i, idx in enumerate(inds):
            img      = np.squeeze(tiff.imread(self.img_files[idx])).astype(np.float32)
            lbl_full = np.load(self.flows_files[idx]).astype(np.float32)  # (nchannels, Z, Y, X)

            if img.ndim == self.dim:
                img = img[np.newaxis]  # (1, Z, Y, X)

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
        imgi   = np.zeros((nimg, self.nchan) + self.tyx, np.float32)
        labels = np.zeros((nimg,) + self.tyx, np.float32)
        links  = [None] * nimg

        for i, idx in enumerate(inds):
            img  = np.squeeze(tiff.imread(self.img_files[idx])).astype(np.float32)
            mask = np.squeeze(tiff.imread(self.mask_files[idx]))
            mask = omnipose.utils.format_labels(mask)

            if img.ndim == self.dim:
                img = img[np.newaxis]

            imgi[i], labels[i], _ = random_crop_warp(
                img=img, Y=mask,
                tyx=self.tyx, v1=self.v1, v2=self.v2,
                nchan=self.nchan, rescale=self.rescale[idx],
                scale_range=self.scale_range,
                gamma_range=self.gamma_range,
                do_flip=self.do_flip, ind=idx,
            )

        out = masks_to_flows_batch(
            labels, links,
            device=self.device,
            omni=self.omni, dim=self.dim,
            affinity_field=self.affinity_field,
        )[:-2]

        X = out[:-1]
        slices = out[-1]
        masks, bd, T, mu = [
            torch.stack([x[(Ellipsis,) + slc] for slc in slices]) for x in X
        ]
        lbl = batch_labels(
            masks, bd, T, mu, self.tyx,
            dim=self.dim, nclasses=self.nclasses,
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


def train_omnipose_3d(
    train_dir,
    model_name='omnipose_3d',
    n_epochs=4000,
    batch_size=1,
    learning_rate=0.005,
    save_every=10,
    weight_decay=1e-5,
    momentum=0.9,
):
    save_dir = os.path.join(train_dir, 'models')
    os.makedirs(save_dir, exist_ok=True)

    all_mask_files = sorted(glob.glob(os.path.join(train_dir, '*_masks.tif')))
    img_files, mask_files, flows_files = [], [], []
    for mf in all_mask_files:
        mask = tiff.imread(mf)
        if np.count_nonzero(mask) / mask.size < 0.01:
            continue
        imf = mf.replace('_masks.tif', '.tif')
        if not os.path.exists(imf):
            continue
        img_files.append(imf)
        mask_files.append(mf)
        flows_files.append(imf.replace('.tif', '_flows3d.npy'))

    ready = [os.path.exists(f) for f in flows_files]
    n_precomputed = sum(ready)
    if n_precomputed == 0:
        raise RuntimeError(
            "No precomputed _flows3d.npy files found. "
            "Submit precompute_flows_3d.sl first."
        )
    if n_precomputed < len(img_files):
        img_files   = [f for f, r in zip(img_files,   ready) if r]
        mask_files  = [f for f, r in zip(mask_files,  ready) if r]
        flows_files = [f for f, r in zip(flows_files, ready) if r]
        print(f"Training on {n_precomputed} tiles with precomputed flows "
              f"({len(ready) - n_precomputed} tiles still pending precompute)")
    else:
        print(f"Training on {len(img_files)} tiles with precomputed flows (fast path)")

    # 10% held-out test split
    rng = random.Random(42)
    indices = list(range(len(img_files)))
    rng.shuffle(indices)
    n_test = max(1, int(0.1 * len(indices)))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    test_img_files   = [img_files[i]   for i in test_idx]
    test_mask_files  = [mask_files[i]  for i in test_idx]
    test_flows_files = [flows_files[i] for i in test_idx]
    img_files   = [img_files[i]   for i in train_idx]
    mask_files  = [mask_files[i]  for i in train_idx]
    flows_files = [flows_files[i] for i in train_idx]
    print(f"  train: {len(img_files)}  test: {len(test_img_files)}")

    model = models.CellposeModel(
        gpu=True, pretrained_model=False,
        nchan=1, nclasses=3, dim=3, omni=True,
    )

    tyx = (128,) * 3

    dataset = FileTrainSet(
        img_files, mask_files,
        flows_files=flows_files,
        rescale=False, diam_train=None, tyx=tyx,
        scale_range=1.0, omni=True, dim=3, nchan=1, nclasses=3,
        device=model.device,
        affinity_field=False,
    )

    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(dataset),
        batch_size=batch_size, drop_last=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        worker_init_fn=dataset.worker_init_fn,
        num_workers=0, pin_memory=False,
    )

    test_dataset = FileTrainSet(
        test_img_files, test_mask_files,
        flows_files=test_flows_files,
        rescale=False, diam_train=None, tyx=tyx,
        scale_range=1.0, omni=True, dim=3, nchan=1, nclasses=3,
        device=model.device,
        affinity_field=False,
    )
    test_sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.SequentialSampler(test_dataset),
        batch_size=batch_size, drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        sampler=test_sampler,
        collate_fn=test_dataset.collate_fn,
        worker_init_fn=test_dataset.worker_init_fn,
        num_workers=0, pin_memory=False,
    )

    LR = _build_lr_schedule(learning_rate, n_epochs)
    model.net.mkldnn = False
    model.autocast = False
    model._set_optimizer(LR[0], momentum, weight_decay, SGD=True)
    model._set_criterion()

    print(f"Starting: {n_epochs} epochs, batch_size={batch_size}, "
          f"lr={learning_rate}, crop={tyx}, device={model.device}")

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
            torch.save(model.net.state_dict(), ckpt)
            print(f"  Saved: {os.path.basename(ckpt)}")

    final = os.path.join(save_dir, model_name)
    torch.save(model.net.state_dict(), final)
    print(f"\nDone. Final model: {final}")
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='File-based 3D omnipose training')
    parser.add_argument('--train_dir', type=str,
                        default='/users/kir-fritzsche/aif490/devel/tissue_analysis/'
                                'segmentation_scripts/for_training/ph3/training_raw')
    parser.add_argument('--model_name',     type=str,   default='omnipose_3d_ph3')
    parser.add_argument('--n_epochs',       type=int,   default=4000)
    parser.add_argument('--batch_size',     type=int,   default=1)
    parser.add_argument('--learning_rate',  type=float, default=0.005)
    parser.add_argument('--save_every',     type=int,   default=10)
    args = parser.parse_args()

    train_omnipose_3d(
        train_dir=args.train_dir,
        model_name=args.model_name,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
    )
