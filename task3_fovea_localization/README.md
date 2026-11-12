# 实验三 眼底图像黄斑中央凹定位

## Kaggle 最终提交

报告中的 `submission (10).csv` 对应提交 **57668363**，公榜 RMSE 为 **11.41176**，私榜为 **12.70134**。X 坐标等权融合 RFM 五折长程、RFM 全量冻结、RFM 全量 LoRA 和 HRNet RGB。Y 坐标按 0.25、0.25、0.50 融合 RFM 短程、VisionFM 短程和 HRNet 亮度增强。

历史七路配方在开发阶段用测试参考比较候选后确定。两个全量分支没有折外预测，训练侧分析使用分组五折结果。[最终配方](configs/kaggle_final_fusion.json)记录七路的训练范围、快照和分轴权重。用原始归档中的七路 `predictions/branches/` 可重建 40 行提交，并校验文件指纹。

~~~bash
python tools/rebuild_submission.py --recipe configs/kaggle_final_fusion.json --branch-dir /path/to/original/predictions/branches --output work/kaggle_final_submission.csv --verify
~~~

## 训练侧五折对照

使用 80 张带坐标的训练图和 20 张待预测图。80 张训练图按原始尺寸分层，并按文件 SHA-256 分组。两张字节相同的训练图放在同一折，五折分别有 15、16、17、16、16 张验证图。

模型先裁去眼底视野以外的黑边，再用冻结的预训练编码器和同一结构的空间解码头定位中心凹。RFM 第 30 轮与 VisionFM 第 20 轮的 EMA 快照在两个坐标轴上均按 1:1 融合。80 张训练图的折外 RMSE 为 **17.36587 px**，单支路分别为 18.64173 和 18.00385 px。归档中的融合 CSV 为按训练折外结果整理的候选，暂无官方线上评测。

## 复算折外误差

需要赛题提供的 80 张训练图的坐标 CSV。将训练标注与 `predictions/oof/` 的归档预测按图像编号对齐后，计算分轴及整体 RMSE。`records/oof_summary.json` 保存复算汇总。

```bash
python tools/evaluate_oof.py --labels data/detection/fovea_localization_train_GT.csv
```

## 重建候选 CSV

Python 3.11 或更新版本可离线运行。脚本从两路归档预测和 40 行模板顺序重建 CSV，并核对输出字节的 SHA-256。

```bash
python tools/rebuild_submission.py --verify
```

## 重新训练

原始五折在 Kaggle Tesla T4 上完成，环境记录为 PyTorch 2.11.0+cu128 和 timm 1.0.29。需要匹配驱动的 CUDA 版 PyTorch 和 torchvision。依赖和上游权重版本见 `requirements.txt` 与 `configs/pretrained_sources.json`。课程数据按以下结构放置。

```text
data/detection/
  train/
  test/
  train_location/
  fovea_localization_train_GT.csv
  sample_submission.csv
```

```bash
python -m pip install -r requirements.txt
python tools/prepare_weights.py --download
python tools/train_branch.py --branch rfm_cv30
python tools/train_branch.py --branch visionfm_cv20
python tools/rebuild_submission.py --branch-dir work/predictions --output work/retrained_submission.csv
```

`--data` 可替换数据目录，`--assets` 可指定校验过的编码器目录，`--dry-run` 会列出训练命令。代码包保留 RFM 80 轮、HRNet RGB 和 HRNet 亮度增强支路供折外比较。RFM 80 轮、HRNet RGB 的原始训练入口以 Kaggle 为运行环境。不同硬件重训时，预测可能有数值差异。

| 文件 | 用途 |
| --- | --- |
| `configs/fold_assignments.csv` | 固定五折及训练图像哈希分组 |
| `configs/fusion.json` | 支路快照及两个坐标轴上的等权融合 |
| `records/` | 训练参数、逐轮日志、检查点指纹和复算汇总 |
| `predictions/oof/` | 五路训练图折外预测 |
| `predictions/branches/` | 重建候选 CSV 所用的模型输出 |
