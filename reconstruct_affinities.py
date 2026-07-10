#!/usr/bin/env python3
"""
Reconstruct full-volume affinity (and fg) maps from per-tile *_aff{c}.tif / *_fg.tif
tiles produced by `affinity_segment.py --save_affinities`.

Generic single-map reconstruction: blends any tile suffix (e.g. _aff0 .. _aff8, _fg)
into a full-volume tif per timepoint, reusing batch_tile_reconstruction's
linear-ramp overlap blending. Lets the affinity->boundary collapse be swept offline
on the reconstructed 9-channel affinities.

Example
-------
  python reconstruct_affinities.py \
      --tile_dir   .../tiles_restored_affinity_raw \
      --output_dir .../tiles_restored_affinity_raw/reconstructed \
      --timepoint  0 \
      --overlap    32
"""
import argparse
import os
import re

import numpy as np
import tifffile as tiff
from natsort import natsorted

# reuse the existing, tested overlap-blend
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_tile_reconstruction import reconstruct_single_channel

DEFAULT_MAPS = [f"aff{c}" for c in range(9)] + ["fg"]


def reconstruct_map(tile_dir, timepoint, suffix, overlap_xy=32, has_timepoints=True):
    """Blend all tiles ending in _{suffix}.tif for `timepoint` into one volume."""
    tif_files = natsorted(f for f in os.listdir(tile_dir) if f.endswith('.tif'))
    end = f"_{suffix}.tif"
    if has_timepoints:
        files = [f for f in tif_files
                 if f"timepoint_{timepoint:04d}" in f and f.endswith(end)]
    else:
        files = [f for f in tif_files if f.endswith(end)]
    if not files:
        raise ValueError(f"No *{end} tiles for timepoint {timepoint} in {tile_dir}")

    tiles = []
    max_z = max_y = max_x = 0
    for f in files:
        az = re.findall(r'_z(\d+)-(\d+)', f)
        ay = re.findall(r'_y(\d+)-(\d+)', f)
        ax = re.findall(r'_x(\d+)-(\d+)', f)
        if not (az and ay and ax):
            print(f"  warning: cannot parse coords from {f}, skipping")
            continue
        zs, ze = int(az[-1][0]), int(az[-1][1])
        ys, ye = int(ay[-1][0]), int(ay[-1][1])
        xs, xe = int(ax[-1][0]), int(ax[-1][1])
        max_z, max_y, max_x = max(max_z, ze), max(max_y, ye), max(max_x, xe)
        data = tiff.imread(os.path.join(tile_dir, f)).astype(np.float32)
        if data.max() > 1.5:
            data /= 65535.0
        tiles.append(dict(z_start=zs, z_end=ze, y_start=ys, y_end=ye,
                          x_start=xs, x_end=xe,
                          original_shape=(ze - zs, ye - ys, xe - xs),
                          data=data))
    image_shape = (max_z, max_y, max_x)
    print(f"  {suffix}: {len(tiles)} tiles -> {image_shape}")
    return reconstruct_single_channel(tiles, image_shape, overlap_xy)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tile_dir",   required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--timepoint",  type=int, default=0)
    p.add_argument("--overlap",    type=int, default=32)
    p.add_argument("--maps", nargs="+", default=DEFAULT_MAPS,
                   help=f"tile suffixes to reconstruct (default: {DEFAULT_MAPS})")
    p.add_argument("--no_timepoints", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for suffix in args.maps:
        vol = reconstruct_map(args.tile_dir, args.timepoint, suffix,
                              overlap_xy=args.overlap,
                              has_timepoints=not args.no_timepoints)
        out = os.path.join(args.output_dir,
                           f"restored_timepoint_{args.timepoint:04d}_{suffix}.tif")
        tiff.imwrite(out, (vol * 65535).astype(np.uint16), compression="zlib")
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
