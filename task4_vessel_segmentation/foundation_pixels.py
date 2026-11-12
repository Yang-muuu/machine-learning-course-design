"""Frozen retinal token caches and outer-fold-only PCA for pixel classification."""
from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


def load_cache(directory, ids):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("scope") != "frozen public foundation encoder; no current labels or learned current statistics"
            or manifest.get("ids") != ids or manifest.get("current_supervised_training") is not False):
        raise ValueError("Require the unmodified public encoder feature cache")
    records = {}
    for key in ids:
        with np.load(directory / f"{key}.npz", allow_pickle=False) as saved:
            if saved["id"].item() != key:
                raise ValueError("Foundation cache image order mismatch")
            feature_map = saved["feature_map"].astype(np.float32)
            if feature_map.shape != (44, 44, 1536) or not np.isfinite(feature_map).all():
                raise ValueError("Unexpected frozen foundation feature lattice")
            records[key] = {"feature_map": feature_map,
                            "original_shape": tuple(saved["original_shape"].tolist()),
                            "padding_top_left": tuple(saved["padding_top_left"].tolist()),
                            "padded_size": int(saved["padded_size"])}
    return records, manifest


def context_features(records, train_ids, output_directory, components=32, threads=6):
    from strong_vessels import cv2
    # INTENT: add retinal semantic context while preserving image-level OOF.
    # REASONING: the dimensionality reducer must not learn validation-image statistics.
    # ASSUMES: cached encoder weights were loaded directly from the original public checkpoint.
    training_tokens = np.concatenate([records[key]["feature_map"].reshape(-1, 1536) for key in train_ids])
    # Layer-normalized token coordinates share their physical units. Centered
    # covariance PCA preserves their learned relative variance; no whitening.
    pca = PCA(n_components=components, svd_solver="randomized", random_state=42)
    with threadpool_limits(limits=threads):
        pca.fit(training_tokens)
        result = {}
        for key, record in records.items():
            feature_map = record["feature_map"]
            reduced = pca.transform(feature_map.reshape(-1, 1536)).reshape(44, 44, components).astype(np.float32)
            padded_size = record["padded_size"]
            dense = cv2.resize(reduced, (padded_size, padded_size), interpolation=cv2.INTER_LINEAR)
            height, width = record["original_shape"]
            top, left = record["padding_top_left"]
            native = dense[top:top+height, left:left+width]
            if native.shape != (height, width, components) or not np.isfinite(native).all():
                raise ValueError("Foundation context did not restore the native image geometry")
            result[key] = np.ascontiguousarray(native.reshape(-1, components))
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pca": pca, "fit_image_ids": train_ids}, output_directory / "foundation_pca.joblib", compress=3)
    report = {"fit_image_ids": train_ids, "components": components, "training_tokens": len(training_tokens),
              "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
              "scaling": "layer-normalized embeddings; centered PCA, no whitening",
              "scope": "PCA fitted only on this outer fold's training image tokens"}
    (output_directory / "foundation_pca.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return result, report


def cache_public_features(args):
    import torch
    from retinal import index_images, write_json
    from strong_vessels import make_model, DEFAULT, cv_preprocess, native_padding
    from retfound_adapter import provenance

    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = dict(DEFAULT, encoder="retfound_green", preprocess="rgb", size=616, patch=616,
               native_grid=True, label_space="native", center_pad=True)
    paths = index_images(Path(args.data) / "train/image")
    ids = [str(i) for i in range(1, 21)]
    if set(paths) != set(ids):
        raise ValueError("Expected the same 20 current training images")
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        raise FileExistsError("Preserve the completed public foundation cache")
    network = make_model(cfg, pretrained=True)
    encoder = network.encoder.eval().requires_grad_(False).to("cuda:0")
    before = next(encoder.parameters()).detach().cpu().clone()
    del network
    for key in ids:
        target = directory / f"{key}.npz"
        if target.exists():
            raise FileExistsError(f"Preserve existing partial cache: {target}")
        image = cv_preprocess(paths[key], cfg)
        from PIL import Image
        with Image.open(paths[key]) as original:
            width, height = original.size
        padding_h, padding_w = native_padding(height, width, cfg)
        if image.shape != (3, 616, 616):
            raise ValueError("Public foundation preprocessing must return CHW RGB")
        x = torch.from_numpy(image).unsqueeze(0).to("cuda:0")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            layers = encoder.get_intermediate_layers(x*2-1, n=[2, 5, 8, 11], reshape=True,
                                                     return_prefix_tokens=False, norm=True)
            feature_map = torch.cat(layers, 1)[0].permute(1, 2, 0).float().cpu().numpy()
        assert feature_map.shape == (44, 44, 1536) and np.isfinite(feature_map).all()
        np.savez_compressed(target, id=np.array(key), feature_map=feature_map,
            original_shape=np.array([height, width]), padding_top_left=np.array([padding_h[0], padding_w[0]]),
            padded_size=np.array(cfg["size"]))
        print("FROZEN_FOUNDATION_CACHED", key, feature_map.shape, flush=True)
    assert torch.equal(before, next(encoder.parameters()).detach().cpu())
    public = provenance()
    public = {key: public[key] for key in ["enum", "url", "sha256", "dataset", "normalization", "license"]}
    write_json(directory / "manifest.json", {"ids": ids,
        "scope": "frozen public foundation encoder; no current labels or learned current statistics",
        "current_supervised_training": False, "encoder_unchanged": True, "public_pretraining": public,
        "layers": [2, 5, 8, 11], "shape_per_image": [44, 44, 1536], "dtype": "float32",
        "normalization": "RGB*2-1", "padding": "center to 616, native 565x584 preserved"})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/segmentation")
    parser.add_argument("--out", default="optimization/frozen_retfound_tokens")
    cache_public_features(parser.parse_args())
