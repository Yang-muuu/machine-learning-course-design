"""Image-held-out pixel boosting with frozen external vessel probabilities."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")

import joblib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from retinal import index_images, write_json
from vessel_model import get_split, read_label, choose_threshold, dice_per_image
from strong_vessels import cv2, fov512

ROOT = Path(__file__).resolve().parent


def features(path, teacher, native=False, multichannel=False):
    """Deterministic per-image features: no labels or learned global statistics."""
    image = Image.open(path).convert("RGB")
    if native:
        teacher = np.asarray(Image.fromarray(teacher).resize(image.size, Image.Resampling.BILINEAR))
    else:
        image = image.resize((512, 512), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32) / 255
    height, width = rgb.shape[:2]
    green = rgb[..., 1]
    fov = ndi.binary_dilation(ndi.binary_fill_holes(ndi.gaussian_filter(green, 1) > .02), iterations=2)
    channels, names = [], []

    def add(name, value):
        channels.append(np.broadcast_to(value, (height, width)).astype(np.float32))
        names.append(name)

    for i, name in enumerate(["red", "green", "blue"]):
        add(name, rgb[..., i])
        add(name + "_ratio", rgb[..., i] / (rgb.sum(-1) + 1e-3))
    lab = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) / 255
    for i, name in enumerate(["L", "a", "b"]):
        add("lab_" + name, lab[..., i])
    add("green_clahe", cv2.createCLAHE(2., (8, 8)).apply((green * 255).astype(np.uint8)) / 255)
    for sigma in [.7, 1.2, 2., 3.5, 6., 12., 24.]:
        mean = ndi.gaussian_filter(green, sigma)
        variance = np.maximum(ndi.gaussian_filter(green * green, sigma) - mean * mean, 0)
        add(f"green_mean_{sigma}", mean)
        add(f"green_dark_contrast_{sigma}", mean - green)
        add(f"green_std_{sigma}", np.sqrt(variance))
        if sigma <= 6:
            dxx = ndi.gaussian_filter(green, sigma, order=(0, 2)) * sigma ** 2
            dyy = ndi.gaussian_filter(green, sigma, order=(2, 0)) * sigma ** 2
            dxy = ndi.gaussian_filter(green, sigma, order=(1, 1)) * sigma ** 2
            delta = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
            add(f"hessian_max_{sigma}", (dxx + dyy + delta) / 2)
            add(f"hessian_min_{sigma}", (dxx + dyy - delta) / 2)
            dx = ndi.gaussian_filter(green, sigma, order=(0, 1))
            dy = ndi.gaussian_filter(green, sigma, order=(1, 0))
            add(f"gradient_magnitude_{sigma}", np.hypot(dx, dy))
    if multichannel:
        for channel, name in [(rgb[..., 0], "red"), (rgb[..., 2], "blue")]:
            for sigma in [.7, 1.5, 3., 6., 12.]:
                mean = ndi.gaussian_filter(channel, sigma)
                variance = np.maximum(ndi.gaussian_filter(channel * channel, sigma) - mean * mean, 0)
                add(f"{name}_mean_{sigma}", mean)
                add(f"{name}_dark_contrast_{sigma}", mean - channel)
                add(f"{name}_std_{sigma}", np.sqrt(variance))
                if sigma <= 6:
                    dxx = ndi.gaussian_filter(channel, sigma, order=(0, 2)) * sigma ** 2
                    dyy = ndi.gaussian_filter(channel, sigma, order=(2, 0)) * sigma ** 2
                    dxy = ndi.gaussian_filter(channel, sigma, order=(1, 1)) * sigma ** 2
                    delta = np.sqrt((dxx-dyy)**2 + 4*dxy**2)
                    add(f"{name}_hessian_max_{sigma}", (dxx+dyy+delta)/2)
                    add(f"{name}_hessian_min_{sigma}", (dxx+dyy-delta)/2)
                    dx = ndi.gaussian_filter(channel, sigma, order=(0, 1))
                    dy = ndi.gaussian_filter(channel, sigma, order=(1, 0))
                    add(f"{name}_gradient_magnitude_{sigma}", np.hypot(dx, dy))
    for length in [9, 17, 25]:
        responses = []
        center = (length - 1) / 2
        for angle in np.arange(0, np.pi, np.pi / 12):
            kernel = np.zeros((length, length), np.uint8)
            offset = np.array([np.cos(angle), np.sin(angle)]) * center
            start = tuple(np.round(center - offset).astype(int))
            end = tuple(np.round(center + offset).astype(int))
            cv2.line(kernel, start, end, 1, 1)
            kernel = kernel.astype(np.float32) / kernel.sum()
            responses.append(cv2.filter2D(green, -1, kernel, borderType=cv2.BORDER_REFLECT))
        responses = np.stack(responses)
        background = cv2.blur(green, (length, length))
        add(f"line_dark_{length}", background - responses.min(0))
        add(f"line_range_{length}", responses.max(0) - responses.min(0))
        add(f"line_std_{length}", responses.std(0))
    p = np.asarray(teacher, dtype=np.float32)
    add("external_vascx_probability", p)
    add("external_vascx_logit", np.log(np.clip(p, 1e-5, 1 - 1e-5) / np.clip(1 - p, 1e-5, 1)) / 10)
    for sigma in [1, 2, 4]:
        add(f"external_vascx_mean_{sigma}", ndi.gaussian_filter(p, sigma))
    add("external_vascx_min3", ndi.minimum_filter(p, 3))
    add("external_vascx_max3", ndi.maximum_filter(p, 3))
    yy, xx = np.mgrid[:height, :width]
    add("x_coordinate", xx / max(1, width - 1))
    add("y_coordinate", yy / max(1, height - 1))
    add("distance_inside_fov", ndi.distance_transform_edt(fov) / max(height, width))
    for i, name in enumerate(["red", "green", "blue"]):
        add("image_mean_" + name, rgb[..., i][fov].mean())
        add("image_std_" + name, rgb[..., i][fov].std())
    result = np.stack(channels, -1).reshape(-1, len(channels))
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite pixel features")
    return result, names


def run(args):
    out = Path(args.out)
    results = out / "results"
    results.mkdir(parents=True, exist_ok=True)
    if (results / "report.json").exists():
        raise FileExistsError("Preserve the completed experiment")
    paths = index_images(Path(args.data) / "train/image")
    labels = index_images(Path(args.data) / "train/label")
    ids = [str(i) for i in range(1, 21)]
    teacher_path = Path(args.teacher)
    teacher_report_path = teacher_path.with_name("report.json")
    teacher_report = json.loads(teacher_report_path.read_text(encoding="utf-8"))
    if "external-only reference" not in teacher_report["scope"]:
        raise ValueError("Require frozen external-only teacher, never cross-fold fitted predictions")
    targets = np.stack([read_label(labels[key], 512) for key in ids]).astype(bool)
    with np.load(teacher_path, allow_pickle=False) as saved:
        if saved["ids"].tolist() != ids or not np.array_equal(saved["targets"], targets):
            raise ValueError("Teacher target/order mismatch")
        teacher = saved["native_probabilities"].astype(np.float32) if args.native and "native_probabilities" in saved else saved["probabilities"].astype(np.float32)
    bank = {}
    start = time.monotonic()
    for key, p in zip(ids, teacher):
        bank[key], names = features(paths[key], p, args.native, args.multichannel)
        print("FEATURES", key, bank[key].shape, "seconds", round(time.monotonic() - start, 1), flush=True)
    if args.outer_teacher_folder:
        if not args.native:
            raise ValueError("Outer teacher features require native pixel alignment")
        names += ["outer_native_probability", "outer_native_mean1", "outer_native_mean2",
                  "outer_native_min3", "outer_native_max3", "outer_native_logit"]
    foundation_records, foundation_manifest = None, None
    if args.foundation_cache:
        if not args.native:
            raise ValueError("Foundation context requires native pixel alignment")
        from foundation_pixels import load_cache
        foundation_records, foundation_manifest = load_cache(args.foundation_cache, ids)
        names += [f"retfound_context_pc{i}" for i in range(args.foundation_components)]
    config = {"name": args.name, "family": "hist_gradient_boosting_pixels", "iterations": args.iterations,
              "max_leaf_nodes": args.leaves, "learning_rate": .08, "min_samples_leaf": 100,
              "l2_regularization": 3., "seed": 42, "sample_positive": 6000, "sample_negative": 24000,
              "n_features": len(names), "feature_names": names, "sklearn": sklearn.__version__,
              "teacher_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
              "teacher_pretraining": teacher_report["pretraining"], "native_grid": args.native,
              "outer_teacher_folder": args.outer_teacher_folder,
              "foundation_cache": args.foundation_cache, "foundation_components": args.foundation_components,
              "foundation_pretraining": foundation_manifest,
              "image_weighting": args.image_weighting,
              "multichannel_derivatives": args.multichannel,
              "prediction_grid": "native probability nearest -> fixed canonical512" if args.native else "canonical512"}
    fitting_targets = {key: (np.asarray(Image.open(labels[key]).convert("L")) >= 128
                            if args.native else targets[ids.index(key)]) for key in ids}
    write_json(results / "run_config.json", config)
    folds, probability_by_id = [], {}
    for fold in (range(5) if args.fold == "all" else [int(args.fold)]):
        train_ids, val_ids = get_split(args.data, fold, 42)
        assert not set(train_ids) & set(val_ids)
        split = {"train_ids": train_ids, "val_ids": val_ids, "seed": 42, "fold": fold}
        fold_bank = bank
        outer_identity = None
        if args.outer_teacher_folder:
            cache_path = Path(args.outer_teacher_folder) / f"fold_{fold}.npz"
            with np.load(cache_path, allow_pickle=False) as saved:
                if saved["ids"].tolist() != ids or saved["source_train_ids"].tolist() != train_ids or saved["source_val_ids"].tolist() != val_ids:
                    raise ValueError("Outer teacher was fitted on different images; reject cross-fold features")
                outer = saved["probabilities_native"].astype(np.float32)
                outer_identity = saved["checkpoint_sha256"].item()
            fold_bank = {}
            for key, probability in zip(ids, outer):
                if probability.shape != fitting_targets[key].shape or not np.isfinite(probability).all():
                    raise ValueError("Outer teacher native geometry mismatch")
                extra = [probability, ndi.gaussian_filter(probability, 1), ndi.gaussian_filter(probability, 2),
                         ndi.minimum_filter(probability, 3), ndi.maximum_filter(probability, 3),
                         np.log(np.clip(probability, 1e-5, 1 - 1e-5) / np.clip(1 - probability, 1e-5, 1)) / 10]
                fold_bank[key] = np.concatenate([bank[key], np.stack(extra, -1).reshape(-1, 6)], 1).astype(np.float32)
        foundation_report = None
        if foundation_records is not None:
            from foundation_pixels import context_features
            context, foundation_report = context_features(foundation_records, train_ids, results / f"fold_{fold}",
                                                          args.foundation_components, args.threads)
            augmented = {}
            for key in ids:
                if context[key].shape[0] != fold_bank[key].shape[0]:
                    raise ValueError("Foundation and local feature geometry mismatch")
                augmented[key] = np.concatenate([fold_bank[key], context[key]], 1).astype(np.float32)
            fold_bank = augmented
            del context
            print("FOUNDATION_CONTEXT_FITTED", fold, foundation_report["explained_variance_ratio_sum"], flush=True)
        rng = np.random.default_rng(42 + fold)
        train_x, train_y, train_weight = [], [], []
        # INTENT: fit pixel rules on 16 complete training images only.
        # REASONING: inverse sampling weights restore each image's actual prior;
        # early_stopping=False avoids an internal random-pixel validation split.
        # ASSUMES: the shared external teacher has never fitted current labels;
        # an optional supervised teacher is recreated per outer split above.
        for key in train_ids:
            target = fitting_targets[key].ravel()
            positive, negative = np.flatnonzero(target), np.flatnonzero(~target)
            pos = rng.choice(positive, min(len(positive), 6000), replace=False)
            neg = rng.choice(negative, min(len(negative), 24000), replace=False)
            selected = np.concatenate([pos, neg])
            train_x.append(fold_bank[key][selected])
            train_y.append(target[selected])
            image_weights = np.concatenate([np.full(len(pos), len(positive) / len(pos)),
                                            np.full(len(neg), len(negative) / len(neg))])
            # Approximate macro-Dice gradient scaling without selecting or
            # dropping any training image; use this fold's training labels only.
            if args.image_weighting == "inverse_vessels":
                image_weights /= max(1, len(positive))
            train_weight.append(image_weights)
        x, y, weight = np.concatenate(train_x), np.concatenate(train_y), np.concatenate(train_weight)
        weight /= weight.mean()
        model = HistGradientBoostingClassifier(max_iter=args.iterations, learning_rate=.08,
            max_leaf_nodes=args.leaves, min_samples_leaf=100, l2_regularization=3.,
            early_stopping=False, random_state=42 + fold)
        directory = results / f"fold_{fold}"
        directory.mkdir(exist_ok=True)
        write_json(directory / "split.json", split)
        fitted = time.monotonic()
        def infer_image(key):
            shape = fitting_targets[key].shape
            prediction = model.predict_proba(fold_bank[key])[:, 1].reshape(shape).astype(np.float32)
            if args.native:
                prediction = np.asarray(Image.fromarray(prediction).resize((512, 512), Image.Resampling.NEAREST))
            return prediction * fov512(paths[key])
        with threadpool_limits(limits=args.threads):
            model.fit(x, y, sample_weight=weight)
            predictions = np.stack([infer_image(key) for key in val_ids]).astype(np.float32)
        for key, prediction in zip(val_ids, predictions):
            probability_by_id[key] = prediction
        val_targets = targets[[ids.index(key) for key in val_ids]]
        threshold, sweep = choose_threshold(predictions, val_targets)
        report = {"fold": fold, "split": split, "mean_vessel_dice": float(dice_per_image(predictions, val_targets, threshold).mean()),
                  "mean_vessel_dice_0_5": float(dice_per_image(predictions, val_targets, .5).mean()),
                  "threshold": threshold, "seconds": time.monotonic() - fitted, "config": config}
        if outer_identity:
            report["outer_teacher_checkpoint_sha256"] = outer_identity
        if foundation_report:
            report["foundation_pca"] = foundation_report
        joblib.dump(model, directory / "model.joblib", compress=3)
        write_json(directory / "report.json", report)
        np.savez_compressed(directory / "oof.npz", ids=np.array(val_ids), probabilities=predictions, targets=val_targets)
        folds.append(report)
        print("PIXEL_FOLD_COMPLETE", fold, report["mean_vessel_dice"], "seconds", report["seconds"], flush=True)
        del x, y, weight, train_x, train_y, train_weight, model, fold_bank
    if args.fold != "all":
        return
    assert set(probability_by_id) == set(ids)
    p = np.stack([probability_by_id[key] for key in ids]).astype(np.float32)
    threshold, sweep = choose_threshold(p, targets)
    report = {"name": args.name, "config": config, "folds": folds, "n_images": 20,
              "mean_vessel_dice": float(dice_per_image(p, targets, threshold).mean()),
              "mean_vessel_dice_0_5": float(dice_per_image(p, targets, .5).mean()), "threshold": threshold,
              "per_image": dict(zip(ids, dice_per_image(p, targets, threshold).tolist())), "threshold_sweep": sweep,
              "seconds": time.monotonic() - start, "scope": "fixed all-20-image five-fold OOF; global threshold is tuning validation",
              "software_reference": "Pedregosa et al., JMLR 12 (2011), scikit-learn"}
    np.savez_compressed(results / "oof_predictions.npz", ids=np.array(ids), probabilities=p, targets=targets)
    write_json(results / "report.json", report)
    print("PIXEL_OOF_COMPLETE", report["mean_vessel_dice"], threshold, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/segmentation")
    parser.add_argument("--teacher", default="optimization/local_vascx_reference/reference_predictions.npz")
    parser.add_argument("--out", default="optimization/local_pixel_hgb_v1")
    parser.add_argument("--name", default="pixel_hgb_v1")
    parser.add_argument("--fold", choices=["all", "0", "1", "2", "3", "4"], default="all")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--leaves", type=int, default=31)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--image-weighting", choices=["uniform", "inverse_vessels"], default="uniform")
    parser.add_argument("--outer-teacher-folder")
    parser.add_argument("--foundation-cache")
    parser.add_argument("--foundation-components", type=int, default=32)
    parser.add_argument("--multichannel", action="store_true")
    run(parser.parse_args())
