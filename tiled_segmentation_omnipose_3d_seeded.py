"""
Native 3D omnipose inference — seeded (2-channel) model.

Input: 2-channel composite tif (2, Z, Y, X) — same format as training composites.
  ch0 : raw signal      — normalised to [0, 1]
  ch1 : nuclear seeds   — binarised to {0, 1}

Output: uint16 label tif (Z, Y, X)
"""
import glob, os, argparse
import numpy as np
import tifffile as tiff
import scipy.ndimage as ndi
import torch

from cellpose_omni import models, io
from omnipose.core import compute_masks
import omnipose.utils

from tiled_segmentation_omnipose_3d import _taper_1d, _tile_weight

io.logger_setup()


def segment_3d(volume, model, tile_size=(128, 128, 128), overlap=16,
               mask_threshold=0.0, flow_threshold=0.4, min_size=5000,
               verbose=True, return_flows=False):
    """
    volume : np.ndarray  (2, Z, Y, X), float32
               ch0 = normalize99(raw), ch1 = binarised seeds
    """
    _, Z, Y, X = volume.shape
    tz, ty, tx = tile_size
    margin = overlap // 2

    step_z = max(1, tz - overlap)
    step_y = max(1, ty - overlap)
    step_x = max(1, tx - overlap)

    dP_acc       = np.zeros((3, Z, Y, X), dtype=np.float32)
    cellprob_acc = np.zeros((Z, Y, X),    dtype=np.float32)
    weight_acc   = np.zeros((Z, Y, X),    dtype=np.float32)

    sz_starts = list(range(0, Z, step_z))
    sy_starts = list(range(0, Y, step_y))
    sx_starts = list(range(0, X, step_x))
    n_tiles = len(sz_starts) * len(sy_starts) * len(sx_starts)

    if verbose:
        print(f"Volume: ({Z},{Y},{X})  tile: {tile_size}  overlap: {overlap}  "
              f"→ {n_tiles} tiles")

    model.net.eval()
    idx = 0
    for sz in sz_starts:
        ez = min(sz + tz, Z)
        sz = max(0, ez - tz)
        for sy in sy_starts:
            ey = min(sy + ty, Y)
            sy = max(0, ey - ty)
            for sx in sx_starts:
                ex = min(sx + tx, X)
                sx = max(0, ex - tx)
                idx += 1

                chunk = volume[:, sz:ez, sy:ey, sx:ex].astype(np.float32)  # (2, tz, ty, tx)

                if verbose:
                    print(f"  [{idx}/{n_tiles}] z={sz}:{ez} y={sy}:{ey} x={sx}:{ex}")

                with torch.no_grad():
                    x_in = torch.from_numpy(chunk[np.newaxis])  # (1, 2, tz, ty, tx)
                    y_pred, _ = model.net(x_in)
                    y = y_pred[0].cpu().numpy()  # (nout, Z, Y, X)

                chunk_dP       = y[:3].astype(np.float32)
                chunk_cellprob = y[3].astype(np.float32)

                cz, cy, cx = ez - sz, ey - sy, ex - sx
                w = _tile_weight(cz, cy, cx, margin)

                dP_acc[:, sz:ez, sy:ey, sx:ex]     += chunk_dP * w
                cellprob_acc[sz:ez, sy:ey, sx:ex]  += chunk_cellprob * w
                weight_acc[sz:ez, sy:ey, sx:ex]    += w

    weight_acc   = np.maximum(weight_acc, 1e-8)
    dP_acc      /= weight_acc
    cellprob_acc /= weight_acc

    cellprob_acc = ndi.gaussian_filter(cellprob_acc, sigma=2)
    dP_acc       = ndi.gaussian_filter(dP_acc,       sigma=(0, 2, 2, 2))

    if verbose:
        print(f"cellprob range: [{cellprob_acc.min():.3f}, {cellprob_acc.max():.3f}]")

    if return_flows:
        return dP_acc, cellprob_acc

    n_above = int((cellprob_acc > mask_threshold).sum())
    if n_above < 100:
        if verbose:
            print(f"Too few above-threshold voxels ({n_above}), returning empty mask.")
        return np.zeros(cellprob_acc.shape, dtype=np.int32)

    print("Computing masks …")
    try:
        masks, _, _, _, _ = compute_masks(
            dP_acc, cellprob_acc,
            flow_threshold=flow_threshold,
            mask_threshold=mask_threshold,
            min_size=min_size,
            do_3D=True,
            cluster=False,
        )
    except (ValueError, RuntimeError) as e:
        print(f"compute_masks failed ({e}), returning empty mask.")
        return np.zeros(cellprob_acc.shape, dtype=np.int32)

    if verbose:
        print(f"Found {int(masks.max())} objects.")

    return masks


def _load_volume(raw_path, masks_path, verbose=True):
    """Load separate raw + seed-mask tifs, return (2, Z, Y, X) normalised volume."""
    raw   = tiff.imread(raw_path).squeeze().astype(np.float32)
    seeds = tiff.imread(masks_path).squeeze().astype(np.float32)
    if raw.ndim != 3:
        raise ValueError(f"Expected 3D raw tif (Z, Y, X), got shape {raw.shape}")
    if seeds.shape != raw.shape:
        raise ValueError(f"Seeds shape {seeds.shape} != raw shape {raw.shape}")
    volume = np.stack([
        omnipose.utils.normalize99(raw),
        (seeds > 0).astype(np.float32),
    ], axis=0)
    if verbose:
        print(f"Volume shape: {volume.shape[1:]}  dtype: {volume.dtype}")
    return volume


def _run_and_save(volume, stem, output_dir, model, tile_size, overlap,
                  mask_threshold, flow_threshold, min_size, verbose, save_flows):
    """Core dispatch: run inference and write outputs; returns output path(s)."""
    os.makedirs(output_dir, exist_ok=True)

    if save_flows:
        dP, cellprob = segment_3d(
            volume, model,
            tile_size=tile_size, overlap=overlap,
            mask_threshold=mask_threshold, flow_threshold=flow_threshold,
            min_size=min_size, verbose=verbose, return_flows=True,
        )
        dp_path       = os.path.join(output_dir, f"{stem}_dP_seeded.tif")
        cellprob_path = os.path.join(output_dir, f"{stem}_cellprob_seeded.tif")
        tiff.imwrite(dp_path,       dP.astype(np.float16))
        tiff.imwrite(cellprob_path, cellprob.astype(np.float16))
        if verbose:
            print(f"Saved flows: {dp_path}")
            print(f"             {cellprob_path}")
        return dp_path, cellprob_path

    masks = segment_3d(
        volume, model,
        tile_size=tile_size, overlap=overlap,
        mask_threshold=mask_threshold, flow_threshold=flow_threshold,
        min_size=min_size, verbose=verbose,
    )
    out_path = os.path.join(output_dir, f"{stem}_masks.tif")
    tiff.imwrite(out_path, masks.astype(np.uint16))
    if verbose:
        print(f"Saved: {out_path}")
    return out_path


def segment_file(input_path, output_dir, model, tile_size=(128, 128, 128),
                 overlap=16, mask_threshold=0.0, flow_threshold=0.4,
                 min_size=5000, verbose=True, save_flows=False):
    """Segment a pre-made 2-channel composite tif (2, Z, Y, X)."""
    stem = os.path.splitext(os.path.basename(input_path))[0]
    if verbose:
        print(f"\n{'='*60}\nInput : {input_path}")

    raw = tiff.imread(input_path).astype(np.float32)
    if raw.ndim != 4 or raw.shape[0] != 2:
        raise ValueError(f"Expected 2-channel composite tif (2, Z, Y, X), got shape {raw.shape}")
    volume = np.stack([
        omnipose.utils.normalize99(raw[0]),
        (raw[1] > 0).astype(np.float32),
    ], axis=0)
    if verbose:
        print(f"Volume shape: {volume.shape[1:]}  dtype: {volume.dtype}")

    return _run_and_save(volume, stem, output_dir, model, tile_size, overlap,
                         mask_threshold, flow_threshold, min_size, verbose, save_flows)


def segment_raw_and_masks(raw_path, masks_path, output_dir, model,
                          tile_size=(128, 128, 128), overlap=16,
                          mask_threshold=0.0, flow_threshold=0.4,
                          min_size=5000, verbose=True, save_flows=False):
    """Segment from separate raw + seed-mask tifs; builds composite in memory."""
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    if verbose:
        print(f"\n{'='*60}\nRaw   : {raw_path}\nMasks : {masks_path}")
    volume = _load_volume(raw_path, masks_path, verbose=verbose)
    return _run_and_save(volume, stem, output_dir, model, tile_size, overlap,
                         mask_threshold, flow_threshold, min_size, verbose, save_flows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Native 3D omnipose seeded-model inference (nchan=2). "
                    "Input modes: (a) --input_file / --input_dir for pre-made composites, "
                    "or (b) --raw + --masks to build the composite in memory.")

    # ── input mode (a): pre-made composite ────────────────────────────────
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--input_file", type=str,
                     help="Single 2-channel composite tif to segment.")
    grp.add_argument("--input_dir",  type=str,
                     help="Directory of composite tifs to segment (all *_composite.tif).")

    # ── input mode (b): separate raw + seed masks ──────────────────────────
    parser.add_argument("--raw",   type=str, default=None,
                        help="Single raw intensity tif (Z, Y, X).")
    parser.add_argument("--masks", type=str, default=None,
                        help="Paired nuclear seed-mask tif (Z, Y, X), same shape as --raw.")

    parser.add_argument("--output_dir",     type=str, required=True)
    parser.add_argument("--model",          type=str, required=True,
                        help="Path to trained seeded omnipose 3D model checkpoint.")
    parser.add_argument("--tile_size",      nargs=3, type=int,
                        default=[128, 128, 128], metavar=("Z", "Y", "X"))
    parser.add_argument("--overlap",        type=int,   default=16)
    parser.add_argument("--mask_threshold", type=float, default=0.0)
    parser.add_argument("--flow_threshold", type=float, default=0.4)
    parser.add_argument("--min_size",       type=int,   default=5000)
    parser.add_argument("--verbose",        action="store_true", default=True)
    parser.add_argument("--gpu",            action="store_true", default=False)
    parser.add_argument("--save_flows",     action="store_true", default=False,
                        help="Save stitched flows (dP + cellprob) as float16 instead of "
                             "computing masks. Outputs {stem}_dP_seeded.tif and "
                             "{stem}_cellprob_seeded.tif.")
    args = parser.parse_args()

    # validate input mode
    use_raw_masks = args.raw is not None or args.masks is not None
    use_composite = args.input_file is not None or args.input_dir is not None
    if use_raw_masks and use_composite:
        parser.error("Use either --raw/--masks or --input_file/--input_dir, not both.")
    if use_raw_masks and not (args.raw and args.masks):
        parser.error("--raw and --masks must both be provided together.")
    if not use_raw_masks and not use_composite:
        parser.error("Provide --raw + --masks, or --input_file / --input_dir.")

    model = models.CellposeModel(
        gpu=args.gpu,
        pretrained_model=False,
        nchan=2, nclasses=3, dim=3, omni=True,
    )
    state_dict = torch.load(args.model, map_location=model.device)
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.net.load_state_dict(state_dict)

    tile_size = tuple(args.tile_size)
    kwargs = dict(
        tile_size=tile_size, overlap=args.overlap,
        mask_threshold=args.mask_threshold, flow_threshold=args.flow_threshold,
        min_size=args.min_size, verbose=args.verbose, save_flows=args.save_flows,
    )

    if use_raw_masks:
        segment_raw_and_masks(args.raw, args.masks, args.output_dir, model, **kwargs)
    else:
        if args.input_file:
            files = [args.input_file]
        else:
            files = sorted(glob.glob(os.path.join(args.input_dir, "*_composite.tif")))
            print(f"Found {len(files)} composite files in {args.input_dir}")
        for f in files:
            segment_file(f, args.output_dir, model, **kwargs)

    print("\nDone.")
