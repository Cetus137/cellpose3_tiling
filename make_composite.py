"""
make_composite.py
-----------------
Combine a raw intensity volume and a nuclear-seed mask volume into a
2-channel composite TIFF for cpSAM (Cellpose 4) segmentation.

Output layout:
    Channel 0 (index 0): raw intensity image
    Channel 1 (index 1): nuclear seed masks
    Shape: (2, Z, Y, X), dtype: float32

Usage
-----
python make_composite.py \\
    --raw   /path/to/sample.tif \\
    --masks /path/to/sample_masks.tif \\
    --output /path/to/output/sample_composite.tif \\
    --verbose
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff


def make_composite(raw_path, masks_path, output_path, verbose=True):
    """
    Read a raw volume and a nuclear-seed mask volume, stack as 2-channel
    composite and write to disk.

    Parameters
    ----------
    raw_path : str or Path
        Path to the raw intensity .tif (shape: Z, Y, X or T, Z, Y, X).
    masks_path : str or Path
        Path to the nuclear seed _masks.tif (same spatial shape as raw).
    output_path : str or Path
        Destination path for the composite .tif.
    verbose : bool
        Print progress messages.

    Returns
    -------
    composite : np.ndarray, shape (2, Z, Y, X), dtype float32
    """
    raw_path    = Path(raw_path)
    masks_path  = Path(masks_path)
    output_path = Path(output_path)

    if verbose:
        print(f"Reading raw:      {raw_path}")
    raw = tiff.imread(str(raw_path))

    if verbose:
        print(f"Reading masks:    {masks_path}")
    masks = tiff.imread(str(masks_path))

    if verbose:
        print(f"  raw   shape (raw): {raw.shape}, dtype: {raw.dtype}")
        print(f"  masks shape (raw): {masks.shape}, dtype: {masks.dtype}")

    # Squeeze out any leading singleton dimensions so both arrays are (Z, Y, X)
    while raw.ndim > 3 and raw.shape[0] == 1:
        raw = raw[0]
    while masks.ndim > 3 and masks.shape[0] == 1:
        masks = masks[0]

    if verbose:
        print(f"  raw   shape (squeezed): {raw.shape}")
        print(f"  masks shape (squeezed): {masks.shape}")

    if raw.shape != masks.shape:
        raise ValueError(
            f"Shape mismatch after squeezing: raw {raw.shape} vs masks {masks.shape}. "
            "Both images must have identical spatial dimensions."
        )

    # Cast to float32 so both channels share a common dtype
    composite = np.stack(
        [raw.astype(np.float32), masks.astype(np.float32)],
        axis=0,                # resulting shape: (2, ...)
    )

    if verbose:
        print(f"Composite shape: {composite.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Writing composite to: {output_path}")
    tiff.imwrite(str(output_path), composite)

    if verbose:
        print("Done.")

    return composite


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Combine a raw .tif and a nuclear-seed _masks.tif into a "
            "2-channel composite for cpSAM segmentation.\n\n"
            "Output shape: (2, Z, Y, X)  —  Ch0=raw, Ch1=nuclear seeds"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw", type=str, required=True,
        help="Path to the raw intensity .tif file.",
    )
    parser.add_argument(
        "--masks", type=str, required=True,
        help="Path to the nuclear-seed _masks.tif file.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for the 2-channel composite .tif.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print progress information.",
    )

    args = parser.parse_args()
    make_composite(args.raw, args.masks, args.output, verbose=args.verbose)
