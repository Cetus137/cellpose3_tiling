"""
Native 3D omnipose inference.

Uses the model in full 3D mode (dim=3, omni=True) — matching the training setup.
Tiles the input volume into (tile_size)^3 chunks (default 128^3, matching the
training crop size), stitches predicted flows with linear-taper weighted averaging,
then runs compute_masks once on the full stitched volume.

Input:  single-channel 3D tif  (Z, Y, X)
Output: uint16 label tif       (Z, Y, X)
"""
import glob, os, argparse
import numpy as np
import tifffile as tiff
import scipy.ndimage as ndi
import torch

from cellpose_omni import models, io
from omnipose.core import compute_masks
import omnipose.utils

io.logger_setup()


def _gamma(img, gamma):
    if gamma == 1.0:
        return img
    return np.power(np.clip(img, 0, None), gamma).astype(np.float32)


# ── cosine-taper weight ────────────────────────────────────────────────────

def _taper_1d(n, margin):
    w = np.ones(n, dtype=np.float32)
    if margin > 0 and n > 2 * margin:
        ramp = np.linspace(0, 1, margin, dtype=np.float32)
        w[:margin]  = ramp
        w[-margin:] = ramp[::-1]
    return w


def _tile_weight(sz, sy, sx, margin):
    wz = _taper_1d(sz, margin)
    wy = _taper_1d(sy, margin)
    wx = _taper_1d(sx, margin)
    return wz[:, None, None] * wy[None, :, None] * wx[None, None, :]


# ── core inference ─────────────────────────────────────────────────────────

def segment_3d(volume, model, tile_size=(128, 128, 128), overlap=16,
               mask_threshold=0.0, flow_threshold=0.4, min_size=5000,
               verbose=True, return_flows=False):
    """
    Tile a 3D volume, run the native 3D omnipose model on each chunk,
    stitch flows, and return the final label mask.

    Parameters
    ----------
    volume : np.ndarray  (Z, Y, X), float32, already normalised to [0, 1]
    model  : cellpose_omni CellposeModel  (nchan=1, nclasses=3, dim=3, omni=True)
    tile_size : (tz, ty, tx) — should match training crop (128^3 by default)
    overlap   : voxels of overlap on each side; cosine taper blends seams
    """
    Z, Y, X = volume.shape
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
        sz = max(0, ez - tz)   # snap to edge so tile is always full-size
        for sy in sy_starts:
            ey = min(sy + ty, Y)
            sy = max(0, ey - ty)
            for sx in sx_starts:
                ex = min(sx + tx, X)
                sx = max(0, ex - tx)
                idx += 1

                chunk = volume[sz:ez, sy:ey, sx:ex].astype(np.float32)

                if verbose:
                    print(f"  [{idx}/{n_tiles}] z={sz}:{ez} y={sy}:{ey} x={sx}:{ex}")

                # ── forward pass ──────────────────────────────────────────
                # Call model.net directly (same path as training loop) to avoid
                # eval() machinery that breaks on CPU with DataParallel wrappers
                # and misroutes 3D chunks through the stacked-2D _run_3D path.
                # Input: (1, 1, Z, Y, X)  Output: y_pred[0] → (nout, Z, Y, X)
                # nout = nclasses+1 = 4: [dz, dy, dx, dist]
                with torch.no_grad():
                    x_in = torch.from_numpy(
                        chunk[np.newaxis, np.newaxis].astype(np.float32)
                    )
                    y_pred, _ = model.net(x_in)
                    y = y_pred[0].cpu().numpy()  # (nout, Z, Y, X)
                chunk_dP       = y[:3].astype(np.float32)  # (3, Z, Y, X) flow vectors
                chunk_cellprob = y[3].astype(np.float32)   # (Z, Y, X) distance/cellprob

                cz, cy, cx = ez - sz, ey - sy, ex - sx
                w = _tile_weight(cz, cy, cx, margin)

                dP_acc[:, sz:ez, sy:ey, sx:ex]     += chunk_dP * w
                cellprob_acc[sz:ez, sy:ey, sx:ex]  += chunk_cellprob * w
                weight_acc[sz:ez, sy:ey, sx:ex]    += w

    # ── normalise by accumulated weights ──────────────────────────────────
    weight_acc   = np.maximum(weight_acc, 1e-8)
    dP_acc      /= weight_acc
    cellprob_acc /= weight_acc

    # ── light smoothing (matches existing omnipose pipeline) ──────────────
    cellprob_acc = ndi.gaussian_filter(cellprob_acc, sigma=2)
    dP_acc       = ndi.gaussian_filter(dP_acc,       sigma=(0, 2, 2, 2))

    if verbose:
        print(f"cellprob range: [{cellprob_acc.min():.3f}, {cellprob_acc.max():.3f}]")

    if return_flows:
        return dP_acc, cellprob_acc

    # Skip compute_masks if almost nothing is above threshold — avoids divide-by-zero
    # NaN crash inside omnipose when the above-threshold region is degenerate.
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


# ── file-level entry point ─────────────────────────────────────────────────

def segment_file(input_path, output_dir, model, tile_size=(128, 128, 128),
                 overlap=16, gamma=1.0, mask_threshold=0.0,
                 flow_threshold=0.4, min_size=5000, verbose=True,
                 save_flows=False):

    os.makedirs(output_dir, exist_ok=True)
    stem   = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{stem}_masks.tif")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Input : {input_path}")
        print(f"Output: {out_path}")

    volume = tiff.imread(input_path).squeeze()
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D (Z,Y,X) tif, got shape {volume.shape}")

    if verbose:
        print(f"Volume shape: {volume.shape}  dtype: {volume.dtype}")

    volume = omnipose.utils.normalize99(volume.astype(np.float32))
    if gamma != 1.0:
        volume = _gamma(volume, gamma)

    if save_flows:
        dP, cellprob = segment_3d(
            volume, model,
            tile_size=tile_size, overlap=overlap,
            mask_threshold=mask_threshold, flow_threshold=flow_threshold,
            min_size=min_size, verbose=verbose, return_flows=True,
        )
        dp_path      = os.path.join(output_dir, f"{stem}_dP_omni.tif")
        cellprob_path = os.path.join(output_dir, f"{stem}_cellprob_omni.tif")
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

    tiff.imwrite(out_path, masks.astype(np.uint16))
    if verbose:
        print(f"Saved: {out_path}")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Native 3D omnipose inference (dim=3, omni=True). "
                    "Input: single-channel 3D tif (Z,Y,X). "
                    "Output: uint16 label tif.")

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input_file", type=str,
                     help="Single 3D tif to segment.")
    grp.add_argument("--input_dir",  type=str,
                     help="Directory of 3D tifs to segment (all *.tif).")

    parser.add_argument("--output_dir",       type=str,   required=True)
    parser.add_argument("--model",            type=str,   required=True,
                        help="Path to trained omnipose 3D model checkpoint.")
    parser.add_argument("--tile_size",        nargs=3, type=int,
                        default=[128, 128, 128], metavar=("Z", "Y", "X"),
                        help="Tile size for inference (default: 128 128 128, "
                             "matching training crop size).")
    parser.add_argument("--overlap",          type=int,   default=16,
                        help="Overlap in voxels on each side (default: 16).")
    parser.add_argument("--gamma",            type=float, default=1.0)
    parser.add_argument("--mask_threshold",   type=float, default=0.0,
                        help="Cell-probability threshold for mask computation.")
    parser.add_argument("--flow_threshold",   type=float, default=0.4)
    parser.add_argument("--min_size",         type=int,   default=5000,
                        help="Minimum object size in voxels.")
    parser.add_argument("--verbose",          action="store_true", default=True)
    parser.add_argument("--gpu",              action="store_true", default=False,
                        help="Use GPU if available (default: CPU).")
    parser.add_argument("--save_flows",       action="store_true", default=False,
                        help="Save stitched flows (dP + cellprob) instead of computing masks. "
                             "Outputs {stem}_dP_omni.tif and {stem}_cellprob_omni.tif.")
    args = parser.parse_args()

    model = models.CellposeModel(
        gpu=args.gpu,
        pretrained_model=False,
        nchan=1, nclasses=3, dim=3, omni=True,
    )
    state_dict = torch.load(args.model, map_location=model.device)
    # cellpose_omni saves net.state_dict() with a 'module.' prefix; strip it
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.net.load_state_dict(state_dict)

    tile_size = tuple(args.tile_size)

    if args.input_file:
        files = [args.input_file]
    else:
        files = sorted(f for f in glob.glob(os.path.join(args.input_dir, "*.tif"))
                       if not f.endswith("_masks.tif"))
        print(f"Found {len(files)} files in {args.input_dir}")

    for f in files:
        segment_file(
            f, args.output_dir, model,
            tile_size=tile_size, overlap=args.overlap,
            gamma=args.gamma, mask_threshold=args.mask_threshold,
            flow_threshold=args.flow_threshold, min_size=args.min_size,
            verbose=args.verbose, save_flows=args.save_flows,
        )

    print("\nDone.")
