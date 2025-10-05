# -*- coding: utf-8 -*-
"""
Chunked threshold for huge rasters (zero-copy approach).
- Reads the raster in windows/blocks to avoid MemoryError
- Auto-detects scale (0–1, 0–100, or 0–10000) from a small sample
- Writes a uint8 mask (1 = >= threshold; 0 = otherwise), tiled + compressed

USAGE (example):
    python raster_threshold_chunked.py --tif input.tif --out mask_40.tif --threshold 0.4
"""
import argparse, sys, os
import numpy as np
import rasterio
from rasterio.windows import Window

def infer_threshold_value(src, threshold):
    """Infer native threshold based on sampled data range.
       If data looks like 0..1 -> use threshold
       If data looks like 0..100 -> use threshold*100
       If data looks like 0..10000 -> use threshold*10000
    """
    # sample a small window from top-left block
    first_window = None
    for _, w in src.block_windows(1):
        first_window = w
        break
    if first_window is None:
        raise RuntimeError("No windows found in the raster.")

    arr = src.read(1, window=first_window, masked=True)
    valid = arr.compressed() if np.ma.isMaskedArray(arr) else arr.flatten()
    # try to get a robust max
    if valid.size == 0:
        data_max = 1.0
    else:
        data_max = float(np.nanpercentile(valid, 99.9))

    if data_max <= 1.0:
        scale = 1.0
    elif data_max <= 110.0:
        scale = 100.0
    elif data_max <= 11000.0:
        scale = 10000.0
    else:
        # fallback: guess no scaling
        scale = 1.0

    thr_native = threshold * scale
    return thr_native, scale, data_max

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True, help="Input GeoTIFF path")
    ap.add_argument("--out", required=True, help="Output mask GeoTIFF path")
    ap.add_argument("--threshold", type=float, default=0.4, help="Threshold as fraction (0.4 = 40%)")
    ap.add_argument("--overview", type=int, default=0, help="Optional: downsample factor for output (e.g., 2, 4). 0 = full res")
    args = ap.parse_args()

    tif_path = args.tif
    out_path = args.out
    thr = args.threshold
    overview = args.overview

    if not os.path.exists(tif_path):
        print("Input not found:", tif_path, file=sys.stderr)
        sys.exit(2)

    with rasterio.open(tif_path) as src:
        # Infer threshold in native units
        thr_native, scale, data_max = infer_threshold_value(src, thr)
        print(f"[INFO] Data max≈{data_max:.2f}, scale_guess={scale}, threshold_native={thr_native:.2f}")

        profile = src.profile.copy()
        # Force single-band uint8 output
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0,
            tiled=True,
            compress="LZW"
        )

        # Handle optional downsample
        if overview and overview > 1:
            new_height = max(1, src.height // overview)
            new_width  = max(1, src.width  // overview)
            profile.update(height=new_height, width=new_width, transform=src.transform * src.transform.scale(overview, overview))
        # NOTE: The above transform update via scale is conceptual; for exact georeferencing,
        # consider using rasterio.warp.calculate_default_transform & reproject.
        # To keep runtime simple, we do a nearest-neighbor block-wise stride read below.

        with rasterio.open(out_path, "w", **profile) as dst:
            # Iterate over blocks/windows of the source
            for ji, w in src.block_windows(1):
                # If downsampling, stride the window
                if overview and overview > 1:
                    # compute a coarse window that samples every Nth pixel
                    row_off = w.row_off
                    col_off = w.col_off
                    height  = (w.height // overview)
                    width   = (w.width  // overview)
                    if height <= 0 or width <= 0:
                        continue
                    # read with out_shape to downsample this block
                    data = src.read(1, window=w, out_shape=(height, width), resampling=rasterio.enums.Resampling.nearest)
                else:
                    data = src.read(1, window=w)

                mask = (data >= thr_native).astype(np.uint8)

                # Destination window has the same shape if not downsampling,
                # otherwise we need to map indices accordingly. For simplicity,
                # write at the same window if not downsampling; if downsampling,
                # compute destination offset/size.
                if overview and overview > 1:
                    dst_window = Window(
                        col_off=w.col_off // overview,
                        row_off=w.row_off // overview,
                        width=mask.shape[1],
                        height=mask.shape[0]
                    )
                    dst.write(mask, 1, window=dst_window)
                else:
                    dst.write(mask, 1, window=w)

    print("[OK] Mask written to:", out_path)

if __name__ == "__main__":
    main()
