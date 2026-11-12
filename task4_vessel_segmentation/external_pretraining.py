"""Load independently trained FIVES initializations into fresh competition folds."""
from pathlib import Path

import torch

FIVES_MD5 = "789c80dd5376a82063e27fa49192bac9"


def validate_initialization(checkpoint, cfg):
    """Keep external pretraining separate from current-fold supervised weights."""
    header = checkpoint.get("external_pretraining", {})
    source = checkpoint.get("config", {})
    if (header.get("dataset") != "FIVES" or header.get("source_md5") != FIVES_MD5
            or header.get("current_competition_used") is not False or header.get("smoke_only") is not False
            or source.get("task") != "external_pretrain"):
        raise ValueError("Require completed external-only FIVES pretraining, never a smoke/current-task checkpoint")
    split = checkpoint["split"]
    train_ids, val_ids = split["train_ids"], split["val_ids"]
    if (len(train_ids) != 480 or len(val_ids) != 120 or len(set(train_ids + val_ids)) != 600
            or not all(key.startswith("FIVES_") for key in train_ids + val_ids)
            or header.get("training_count") != len(train_ids) or header.get("validation_count") != len(val_ids)):
        raise ValueError("FIVES source must use the declared 480/120 image split")
    for key, default in [("encoder", None), ("cnn_skip", None), ("decoder_norm", "batch")]:
        if source.get(key, default) != cfg.get(key, default):
            raise ValueError(f"External initialization architecture differs: {key}")
    if cfg.get("preprocess") != source.get("preprocess"):
        raise ValueError("The dense FIVES initialization requires its declared input transform")
    return header


def load_initialization(model, cfg, fold):
    # INTENT: each outer fold starts fresh from external data only.
    # REASONING: sharing a current-task trained fold would leak held-out labels.
    # ASSUMES: two external folds are selected by fixed fold parity, never by current validation Dice.
    paths = cfg["external_init_paths"]
    if len(paths) != 2:
        raise ValueError("Expected the two completed external initializations")
    source_fold = int(fold) % 2
    path = Path(paths[source_fold])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    header = validate_initialization(checkpoint, cfg)
    if int(checkpoint.get("fold", -1)) != source_fold or int(checkpoint["split"]["fold"]) != source_fold:
        raise ValueError("External initialization order differs from the declared parity mapping")
    model.load_state_dict(checkpoint["model"], strict=True)
    return {"enum": "external FIVES dense vessel pretraining", "initialization_path": str(path),
            "source_name": checkpoint["config"]["name"], "source_fold": source_fold,
            "source_epoch": checkpoint["epoch"], "external_pretraining": header,
            "public_initialization": checkpoint.get("pretrained_source"),
            "current_fold": int(fold), "selection": "fixed source_fold=current_fold%2"}
