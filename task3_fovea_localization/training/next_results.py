"""Replay checkpoints, verify held-out predictions, and report the three trials."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch

from next_models import (ImageBank, MODELS, file_hash, load_saved, predict, score,
                            seed_everything, state_hash)


def write_submission(sample_path, predictions, output, identity=False):
    sample = pd.read_csv(sample_path)
    assert list(sample.columns) == ["ImageID", "value"] and sample.ImageID.is_unique
    indexed = predictions.set_index("image_id")
    prefix = "identity_" if identity else "pred_"
    values = []
    for key in sample.ImageID:
        match = re.fullmatch(r"(\d+)_Fovea_([XY])", str(key))
        assert match is not None
        values.append(float(indexed.loc[int(match[1]), prefix + match[2].lower()]))
    sample["value"] = values
    assert len(sample) == 40 and np.isfinite(values).all()
    sample.to_csv(output, index=False, float_format="%.8f")


def overlay(frame, root, split, destination, ncols=4):
    width, height = 400, 315
    canvas = Image.new("RGB", (width*ncols, height*((len(frame)+ncols-1)//ncols)), "#111827")
    for i, row in enumerate(frame.itertuples(index=False)):
        with Image.open(Path(root) / split / row.filename) as source:
            image = source.convert("RGB")
        scale = min(width / image.width, 275 / image.height)
        image = image.resize((round(image.width*scale), round(image.height*scale)))
        draw = ImageDraw.Draw(image)
        if hasattr(row, "x"):
            x, y = row.x*scale, row.y*scale
            draw.ellipse((x-5,y-5,x+5,y+5),outline="#22ef9a",width=2)
        x, y = row.pred_x*scale, row.pred_y*scale
        draw.line((x-6,y,x+6,y),fill="#ff4466",width=2)
        draw.line((x,y-6,x,y+6),fill="#ff4466",width=2)
        ox, oy = (i % ncols)*width, (i // ncols)*height
        canvas.paste(image,(ox,oy))
        label = row.filename + (f"  error={row.distance_px:.1f}px" if hasattr(row,"distance_px") else "")
        ImageDraw.Draw(canvas).text((ox+6,oy+282),label,fill="white")
    canvas.save(destination)


def run(args):
    assert Path("/kaggle/working").is_dir() and torch.cuda.is_available()
    seed_everything(20261021)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    root = args.output
    config = json.loads((root/"config.json").read_text())
    smoke = json.loads((root/"gpu_smoke_checks.json").read_text())
    train = pd.read_csv(args.artifacts/"train.csv")
    test = pd.read_csv(args.artifacts/"test.csv")
    train_bank = ImageBank(train,args.data,"train",device)
    test_bank = ImageBank(test,args.data,"test",device)
    assert file_hash(args.artifacts/"train.csv") == config["train_manifest_sha256"]
    assert train.groupby("group").fold.nunique().eq(1).all()
    raw = pd.read_csv(args.data/"fovea_localization_train_GT.csv").rename(
        columns={"data":"image_id","Fovea_X":"x","Fovea_Y":"y"}).set_index("image_id")
    np.testing.assert_allclose(train[["x","y"]],raw.loc[train.image_id][["x","y"]],atol=1e-10,rtol=0)
    summaries, checkpoints = {}, []
    all_predictions, all_test = [], []
    fig, axes = plt.subplots(1,len(MODELS),figsize=(5*len(MODELS),4),constrained_layout=True)
    for ax, name in zip(axes, MODELS):
        fold_oof, fold_test, fold_metrics = [], [], []
        for fold in range(5):
            folder = root/name/f"fold{fold}"
            manifest = json.loads((folder/"training_manifest.json").read_text())
            assert file_hash(folder/"model.pt") == manifest["checkpoint_sha256"]
            history = pd.read_csv(folder/"history.csv")
            assert history.epoch.tolist() == list(range(1,81))
            assert np.isfinite(history[["loss","gradient_norm","lr"]]).all().all()
            assert manifest["actual_head_update"] and manifest["frozen_encoder_unchanged"]
            valid_ids = set(train.loc[train.fold == fold,"image_id"])
            train_ids = set(train.image_id)-valid_ids
            assert set(manifest["train_ids"]) == train_ids
            assert set(manifest["validation_ids"]) == valid_ids
            assert set(train.set_index("image_id").loc[list(train_ids),"group"]).isdisjoint(
                       train.set_index("image_id").loc[list(valid_ids),"group"])
            model, saved = load_saved(folder/"model.pt",args.assets,device)
            assert saved["completed_epochs"] == 80 and saved["fold"] == fold
            assert saved["train_ids"] == manifest["train_ids"]
            assert saved["validation_ids"] == manifest["validation_ids"]
            assert state_hash(model.encoder) == smoke[name]["frozen_encoder_sha256"]
            assert state_hash(model.head) != saved["initial_head_sha256"]
            assert saved["source_file_sha256"] == model.encoder.provenance["sha256"]
            va = np.flatnonzero(train.fold.values == fold)
            oi, ot = predict(model,train_bank,va)
            ti, tt = predict(model,test_bank,np.arange(len(test)))
            oof = pd.read_csv(folder/"oof.csv")
            tests = pd.read_csv(folder/"test.csv")
            assert oof.image_id.tolist() == train.iloc[va].image_id.tolist()
            assert oof.fold.eq(fold).all()
            np.testing.assert_allclose(oof[["pred_x","pred_y"]],ot,atol=.003,rtol=0)
            np.testing.assert_allclose(oof[["identity_x","identity_y"]],oi,atol=.003,rtol=0)
            np.testing.assert_allclose(tests[["pred_x","pred_y"]],tt,atol=.003,rtol=0)
            np.testing.assert_allclose(tests[["identity_x","identity_y"]],ti,atol=.003,rtol=0)
            result = score(oof[["x","y"]],oof[["pred_x","pred_y"]])
            assert abs(result["rmse_pixels"]-history.val_rmse.iloc[-1]) < .003
            fold_metrics.append(dict(fold=fold,**result))
            oof["model_name"], tests["model_name"] = name, name
            fold_oof.append(oof)
            fold_test.append(tests)
            ax.plot(history.epoch,history.val_rmse,marker=".",label=f"fold {fold}")
            checkpoints.append(dict(model=name,fold=fold,checkpoint_sha256=manifest["checkpoint_sha256"],
                                    epochs=80,train_count=len(train_ids),validation_count=len(valid_ids),
                                    frozen_encoder_unchanged=True,checkpoint_prediction_replay=True))
            del model,saved
            gc.collect()
            torch.cuda.empty_cache()
            print(f"REPLAY PASS {name} fold={fold}: {result['rmse_pixels']:.4f}px",flush=True)
        oof = pd.concat(fold_oof).sort_values("image_id").reset_index(drop=True)
        tests = pd.concat(fold_test)
        assert len(oof) == 80 and oof.image_id.is_unique
        assert len(tests) == 100 and tests.groupby("image_id").size().eq(5).all()
        assert oof.group.tolist() == train.set_index("image_id").loc[oof.image_id].group.tolist()
        oof.to_csv(root/name/"oof.csv",index=False)
        tests.to_csv(root/name/"test_all_folds.csv",index=False)
        means = tests.groupby("image_id")[["pred_x","pred_y","identity_x","identity_y"]].mean()
        prediction = test.set_index("image_id").join(means).reset_index()
        prediction.to_csv(root/name/"test_predictions.csv",index=False)
        write_submission(args.data/"sample_submission.csv",prediction,root/f"submission_{name}.csv")
        write_submission(args.data/"sample_submission.csv",prediction,root/f"submission_{name}_identity.csv",True)
        summaries[name] = dict(oof=score(oof[["x","y"]],oof[["pred_x","pred_y"]]),
                               identity_oof=score(oof[["x","y"]],oof[["identity_x","identity_y"]]),
                               unique_79_oof=score(oof.drop_duplicates("group")[["x","y"]],
                                                   oof.drop_duplicates("group")[["pred_x","pred_y"]]),
                               fold_metrics=fold_metrics,
                               head_parameters=smoke[name]["head_parameters"],
                               submission_sha256=file_hash(root/f"submission_{name}.csv"))
        overlay(oof.nlargest(12,"distance_px"),args.data,"train",root/name/"worst12_oof.jpg")
        overlay(prediction,args.data,"test",root/name/"test_overlay.jpg")
        ax.set(title=name,xlabel="Epoch",ylabel="Held-out coordinate RMSE / px")
        ax.legend(fontsize=8)
        all_predictions.append(oof)
        all_test.append(tests)
    fig.savefig(root/"validation_curves.png",dpi=140)
    plt.close(fig)
    pd.concat(all_predictions).to_csv(root/"oof_all_models.csv",index=False)
    pd.concat(all_test).to_csv(root/"test_all_models.csv",index=False)
    # A predeclared equal-weight ensemble is diagnostic; no weight search on 80 examples.
    ensemble_oof = pd.concat(all_predictions).groupby("image_id")[["pred_x","pred_y"]].mean()
    ensemble_oof = train.set_index("image_id").join(ensemble_oof).reset_index()
    ensemble_oof["distance_px"] = np.linalg.norm(ensemble_oof[["x","y"]].values-
                                               ensemble_oof[["pred_x","pred_y"]].values,axis=1)
    ensemble_oof.to_csv(root/"oof_equal_ensemble.csv",index=False)
    means = pd.concat(all_test).groupby("image_id")[["pred_x","pred_y"]].mean()
    ensemble_test = test.set_index("image_id").join(means).reset_index()
    ensemble_test.to_csv(root/"test_equal_ensemble.csv",index=False)
    write_submission(args.data/"sample_submission.csv",ensemble_test,root/"submission_equal_ensemble.csv")
    metrics = dict(models=summaries,equal_ensemble=score(ensemble_oof[["x","y"]],
                                                       ensemble_oof[["pred_x","pred_y"]]),
                   model_count=20,folds=5,repeats=1,epochs_per_model=80,
                   total_logged_epochs=1600,independent_training_images=79,
                   absence_detection_validated=False,repeat_rmse_std=None,
                   primary="All 80 original-pixel held-out predictions, fixed horizontal-flip TTA",
                   evaluation="development cross-validation; not a competition leaderboard score",
                   kaggle_submission_performed=False)
    (root/"metrics.json").write_text(json.dumps(metrics,indent=2))
    verification = dict(status="PASS",checkpoint_count=20,total_logged_epochs=1600,
                        original_training_labels_verified=True,source_weights_strictly_loaded=True,
                        frozen_encoders_match_initial_weights=True,checkpoints=checkpoints,
                        all_heldout_predictions_replayed=True,all_test_predictions_replayed=True,
                        replay_tolerance_pixels=.003,kaggle_submission_performed=False)
    (root/"checkpoint_verification.json").write_text(json.dumps(verification,indent=2))
    rows = "\n".join(f"| {n} | {summaries[n]['oof']['rmse_pixels']:.4f} | "
                     f"{summaries[n]['identity_oof']['rmse_pixels']:.4f} | {summaries[n]['head_parameters']:,} |"
                     for n in MODELS)
    report = f"""四个新增预训练编码器：真实 Kaggle 分组五折结果

DINOv3 ConvNeXt-Tiny、FLAIR ResNet50、RFM 九层编码器、VisionFM Fundus。
20 个正式模型，各固定 80 epoch，共 1600 行训练历史；完整训练均在 Kaggle GPU。
RETFound-MAE CFP 和 RETFound-DINOv2 Large 的官方权重访问受限，未运行，详见 source_provenance.json。

| 编码器 | 固定翻转 TTA OOF RMSE ↓ | 无翻转 OOF RMSE | 可训练参数 |
|---|---:|---:|---:|
{rows}

预先固定的四模型等权融合 OOF RMSE：{metrics['equal_ensemble']['rmse_pixels']:.4f}。
未搜索融合比例；不得用此开发分数视为独立测试成绩。

同一原分组五折、448×448 视野、224×224 定位热图及区域辅助监督，使用原始附件训练坐标和XML框。
编码器完全冻结并保持 eval 模式；定位头结构相同，输入投影通道随编码器变化。
RFM 与 VisionFM 输入通道相同，定位头初始参数相同。
归一化分别遵循对应权重：FLAIR 像素缩放到0–1、VisionFM为其官方Fundus统计量。
RFM严格使用作者发布的九层权重，不补随机初始化的后续层；VisionFM使用teacher的完整预训练编码器。
视觉语言模型只保留图像编码器的空间特征；没有把全局分类相似度当成坐标。

每个正式模型都从官方初始编码器和随机定位头独立开始；丢弃GPU检查模型。
固定最后第80轮定位头EMA，没有挑选验证最低epoch。全部20个checkpoint严格重载、复算OOF与测试预测。
重复图0039/0077保持同折；每张训练图恰有一次持出预测；OOF分数在所有80张原图坐标上统一计算。
本次流程未读取另行提供的测试参考答案，也未使用其本地公榜/私榜分数进行训练或选型。
没有向比赛提交，未改变原有推荐CSV。完整模型权重保留在Kaggle Output。

仅一次五折、未估计重复间标准差；80张训练图均可见，未验证不可见检测能力。
本次结果只代表冻结编码器的适配方式，不代表这些预训练模型的所有微调方法。
"""
    (root/"RESULTS.md").write_text(report,encoding="utf-8")
    print(report,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--artifacts",type=Path,default=Path("artifacts"))
    parser.add_argument("--assets",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("next_run"))
    run(parser.parse_args())
