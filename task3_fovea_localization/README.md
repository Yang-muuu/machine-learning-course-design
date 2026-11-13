# 实验三 眼底图像黄斑中央凹定位

## Kaggle 最终提交

报告中的 `submission (3).csv`（见仓库根目录 `submissions/submission (3).csv`）对应七支路融合方案最终提交，公榜 RMSE 为 **11.41176**，私榜为 **12.70134**。X 坐标等权融合 `rfm_cv80`、`rfm_full80`、`rfm_lora80` 和 `hrnet_rgb80`（各 0.25）；Y 坐标按 `0.25 / 0.25 / 0.50` 融合 `rfm_cv30`、`visionfm_cv20` 和 `hrnet_luma40`。

仓库已包含 7 个支路的归档原生坐标预测（`predictions/branches/*.csv`）与最终融合配方（`configs/fusion.json`、`configs/kaggle_final_fusion.json`）。离线运行以下命令可从 7 个支路按 `configs/submission_order.csv` 顺序重建 40 行提交文件，并校验与 `predictions/submission.csv` 及正式提交逐字节一致的 SHA-256（`e5a8f5a495b2c23479630b76d0621fa9905cb5a938e65b0b087e38215e90ed12`）：

```bash
python tools/rebuild_submission.py --verify
```

## 训练侧五折对照与折外复算

使用 80 张带坐标的训练图和 20 张待预测图。80 张训练图按原始尺寸分层，并按文件 SHA-256 分组。两张字节相同的训练图放在同一折，五折分别有 15、16、17、16、16 张验证图（`configs/fold_assignments.csv`）。

5 个五折交叉验证支路在 `predictions/oof/` 下保留完整折外预测；`rfm_full80` 与 `rfm_lora80` 使用全部 80 张训练图拟合，不产生折外指标。仅按 OOF 选择的 `rfm_cv30` 与 `visionfm_cv20` 1:1 融合候选配方保存在 `configs/oof_candidate_fusion.json`（OOF RMSE **17.36587 px**），最终融合的 Y 轴五折 OOF RMSE 为 **12.20757 px**。

放入官方 80 张训练坐标文件后可复算全部五折支路、等权 OOF 候选与最终 Y 轴融合的折外误差（对照 `records/oof_summary.json`）：

```bash
python tools/evaluate_oof.py --labels data/detection/fovea_localization_train_GT.csv
python tools/rebuild_submission.py --recipe configs/oof_candidate_fusion.json --output work/oof_candidate_submission.csv --verify
```

## 从头重新训练全部 7 个支路

原始五折在 Kaggle Tesla T4 上完成，两个全量分支在 RTX 4060 Laptop GPU 上完成，环境记录为 PyTorch `2.11.0+cu128` 和 `timm 1.0.29`。先准备匹配驱动的 CUDA 版 PyTorch 与 torchvision，并按 `configs/pretrained_sources.json` 准备公开编码器权重与课程数据：

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
python tools/train_branch.py --branch rfm_cv80
python tools/train_branch.py --branch hrnet_rgb80
python tools/train_branch.py --branch rfm_cv30
python tools/train_branch.py --branch visionfm_cv20
python tools/train_branch.py --branch hrnet_luma40
python tools/train_branch.py --branch rfm_full80
python tools/train_branch.py --branch rfm_lora80
python tools/rebuild_submission.py --branch-dir work/predictions --output work/retrained_submission.csv
```

`rfm_cv80` 和 `hrnet_rgb80` 保留其原始 Kaggle 训练入口（需存在 `/kaggle/input` 与 `/kaggle/working`）；其余支路可在本地 CUDA 环境运行。可用 `--data` 指定数据目录、`--assets` 指定编码器目录、`--dry-run` 查看具体训练命令与参数。不同硬件与软件版本重训时，输出预测可能存在微小数值差异；正式提交的逐字节复现以归档支路预测重建为准。

| 文件 | 用途 |
| --- | --- |
| `configs/fold_assignments.csv` | 固定五折及训练图像哈希分组 |
| `configs/fusion.json` / `configs/kaggle_final_fusion.json` | 最终 7 支路 X/Y 分轴融合配置 |
| `configs/oof_candidate_fusion.json` | R30+V20 等权训练侧 OOF 候选配置 |
| `records/` | 7 个支路的训练参数、逐轮日志、检查点指纹与 Kaggle 提交记录 |
| `predictions/oof/` | 5 个交叉验证支路的训练集折外预测 |
| `predictions/branches/` | 参与最终融合的 7 个支路测试图原生坐标预测 |
| `predictions/submission.csv` | 最终提交文件（与根目录 `submissions/submission (3).csv` 逐字节一致） |
