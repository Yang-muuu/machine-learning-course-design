"""One explicit vessel target and output protocol for the supplied converter.

The original converter resizes an 8-bit L image with PIL BILINEAR, then
encodes every nonzero pixel. Hidden Kaggle labels are unavailable: this is
the user-authorized local proxy, not a claim to have inspected hidden GT.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SCRIPT_PROTOCOL = "supplied_converter_bilinear512_nonzero_v1"
LEGACY_PROTOCOL = "legacy_nearest512_ge128"
FIVEFOLD = "fivefold_seed42"
LOO = "loo20_fixed"
THRESHOLDS = np.round(np.arange(.2, .8001, .025), 3).astype(np.float32)


def read_target(path, protocol=SCRIPT_PROTOCOL):
    image = Image.open(path).convert("L")
    if protocol == SCRIPT_PROTOCOL:
        return np.asarray(image.resize((512, 512), Image.Resampling.BILINEAR)) > 0
    if protocol == LEGACY_PROTOCOL:
        return np.asarray(image.resize((512, 512), Image.Resampling.NEAREST)) >= 128
    raise ValueError(f"Unknown target protocol: {protocol}")


def read_targets(root, ids, protocol=SCRIPT_PROTOCOL):
    from retinal import index_images
    labels = index_images(Path(root) / "train/label")
    return np.stack([read_target(labels[key], protocol) for key in ids])


def probability_transform(p, postprocess="none"):
    if postprocess == "none":
        return np.asarray(p)
    if postprocess == "cross1":
        # max(p) >= t is exactly dilation(p >= t), including image boundaries.
        footprint = ndimage.generate_binary_structure(2, 1)[None]
        return ndimage.maximum_filter(p, footprint=footprint, mode="constant", cval=0)
    raise ValueError(f"Unknown postprocess: {postprocess}")


def binary_predictions(p, threshold, postprocess="none"):
    return probability_transform(p, postprocess) >= threshold


def dice(prediction, targets):
    prediction, targets = np.asarray(prediction, dtype=bool), np.asarray(targets, dtype=bool)
    if prediction.shape != targets.shape or prediction.ndim != 3:
        raise ValueError("Require matching N,H,W predictions and targets")
    intersection = (prediction & targets).sum((1, 2), dtype=np.int64)
    denominator = prediction.sum((1, 2), dtype=np.int64) + targets.sum((1, 2), dtype=np.int64)
    return np.divide(2 * intersection, denominator, out=np.ones(len(targets), dtype=np.float64), where=denominator > 0)


def score(p, targets, threshold=.5, postprocess="none"):
    return dice(binary_predictions(p, threshold, postprocess), targets)


def select_output(p, targets, postprocesses=("none", "cross1")):
    """Exact global threshold search using histograms; no per-image fitting."""
    if p.shape != targets.shape or not np.isfinite(p).all():
        raise ValueError("Invalid evaluation probabilities/targets")
    best = None
    for postprocess in postprocesses:
        values = probability_transform(p, postprocess)
        all_scores = []
        for image, target in zip(values, targets):
            # Bucket j contains thresholds[ j-1 ] <= p < thresholds[ j ].
            bucket = np.searchsorted(THRESHOLDS, image.ravel(), side="right")
            total = np.bincount(bucket, minlength=len(THRESHOLDS) + 1)
            positive = np.bincount(bucket[target.ravel()], minlength=len(THRESHOLDS) + 1)
            predicted = total[::-1].cumsum()[::-1][1:]
            tp = positive[::-1].cumsum()[::-1][1:]
            denominator = predicted + int(target.sum())
            all_scores.append(np.divide(2 * tp, denominator, out=np.ones(len(THRESHOLDS)), where=denominator > 0))
        all_scores = np.stack(all_scores)
        means = all_scores.mean(0)
        index = int(means.argmax())
        candidate = {"mean_vessel_dice": float(means[index]), "threshold": round(float(THRESHOLDS[index]), 3),
                     "postprocess": postprocess, "per_image_scores": all_scores[:, index].tolist()}
        if best is None or candidate["mean_vessel_dice"] > best["mean_vessel_dice"]:
            best = candidate
    return best


def split_for(root, fold, cfg):
    from vessel_model import get_split
    from retinal import image_paths
    fold = int(fold)
    if cfg.get("cv_protocol", FIVEFOLD) == FIVEFOLD:
        if not 0 <= fold < 5:
            raise ValueError("Fivefold index outside 0..4")
        return get_split(root, fold, cfg["seed"])
    if cfg.get("cv_protocol") != LOO or cfg.get("checkpoint_selection") != "fixed_final_ema":
        raise ValueError("LOO requires a declared fixed final EMA checkpoint")
    ids = [path.stem for path in image_paths(Path(root) / "train/image")]
    if ids != [str(i) for i in range(1, 21)] or not 0 <= fold < 20:
        raise ValueError("LOO requires the complete ordered 20-image dataset")
    return [key for key in ids if key != ids[fold]], [ids[fold]]
