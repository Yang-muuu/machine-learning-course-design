"""Threshold in native pixels, then apply the supplied 8-bit resize semantics.

The output is a grid-quantized confidence field, not a calibrated probability.
For every declared threshold its binary mask is exactly the converter result
for the native binary mask. This transform never reads labels.
"""
import numpy as np
from PIL import Image
from competition_protocol import THRESHOLDS

RASTER_GRID = np.unique(np.concatenate([
    np.arange(1, 100, dtype=np.float64) / 100., THRESHOLDS.astype(np.float64)
]).astype(np.float32))


def rasterize(binary):
    image = Image.fromarray(np.where(binary, 255, 0).astype(np.uint8))
    return np.asarray(image.resize((512, 512), Image.Resampling.BILINEAR)) > 0


def confidence_field(native_probability, canonical_fov=None):
    if native_probability.ndim != 2 or not np.isfinite(native_probability).all():
        raise ValueError("Require a finite native H,W confidence field")
    result = np.zeros((512, 512), dtype=np.float32)
    for threshold in RASTER_GRID:
        result[rasterize(native_probability >= threshold)] = threshold
    if canonical_fov is not None:
        result *= canonical_fov
    # This identity is exact at every threshold used for output selection.
    for threshold in THRESHOLDS:
        expected = rasterize(native_probability >= threshold)
        if canonical_fov is not None:
            expected &= canonical_fov
        if not np.array_equal(result >= threshold, expected):
            raise AssertionError("Raster confidence does not reproduce supplied-converter threshold semantics")
    return result
