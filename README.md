# 机器学习课程设计（四项实验完整代码与复现记录仓库）

本仓库包含四项计算机视觉课程设计实验的完整训练源码、冻结实验配置、逐轮训练/验证日志及轻量级离线复核脚本，对应课程设计报告第 1–4 章及 Kaggle 最终提交记录。

## 一、Kaggle 最终提交与线上成绩

2026 年 11 月 12 日核对 Kaggle 提交历史与各任务最终提交文件，四份正式提交 CSV 已保存在 [`submissions/`](./submissions/) 目录，完整指纹与时间戳记录保存在 [`kaggle_submissions.json`](./kaggle_submissions.json)。

| 实验 | 目录 | 提交文件 | 评价指标 | 公榜 | 私榜 | 提交 CSV SHA-256 |
| --- | --- | --- | --- | ---: | ---: | --- |
| 1. 植物幼苗图像分类 | [`task1_plant_seedling/`](./task1_plant_seedling/) | [`submissions/submission (1).csv`](./submissions/submission%20(1).csv) | F1 Score | **0.9708** | **0.9708** | `3b5f222fc11e9ebd36c9974e1ab188d8f95dd2862122434cb9728b0865268e62` |
| 2. 人脸表情分类 | [`task2_facial_expression/`](./task2_facial_expression/) | [`submissions/submission (2).csv`](./submissions/submission%20(2).csv) | Accuracy | **0.747** | **0.747** | `cfff4a03b1358d588b8c30eb8bedbff96a5468656f214d571aa7c1a4ada33511` |
| 3. 黄斑中央凹定位 | [`task3_fovea_localization/`](./task3_fovea_localization/) | [`submissions/submission (3).csv`](./submissions/submission%20(3).csv) | RMSE | **11.41176** | **12.70134** | `e5a8f5a495b2c23479630b76d0621fa9905cb5a938e65b0b087e38215e90ed12` |
| 4. 眼底血管分割 | [`task4_vessel_segmentation/`](./task4_vessel_segmentation/) | [`submissions/submission (4).csv`](./submissions/submission%20(4).csv) | Dice | **0.86267** | **0.86530** | `4d516ae13d88d356dfe5ecc9a17f4f6609b1383c469c7de92b6e809ef2bf5d32` |

## 二、训练侧核心方案与五折折外（OOF）记录

| 实验 | 核心架构与训练方案 | 训练侧验证（OOF）结果 |
| --- | --- | --- |
| [实验一](./task1_plant_seedling/README.md) | `ConvNeXt-Small` ($448\times 448$, IN22K→1K)，分层 5 折交叉验证、Mixup/CutMix、步数预热 EMA/Raw 择优；另附 208 张软伪标签对照 | 500 图 OOF 正确 485 张：Acc **0.9700**，Macro-F1 **0.97010** |
| [实验二](./task2_facial_expression/README.md) | 像素 SHA-256 审计去重与同图同折划分；7 组五折架构（`ConvNeXt-Tiny+ECA`、`DINOv2-S/14`、`DDAMFN++` [Drop0.3/0.2/SAM]、`DAN`、`EmotiEffNet-B2`）；方案 A 四模型加权 / 方案 B 七模型 42 维对数概率 L2 逻辑回归堆叠 | 28,165 图 OOF：方案 A Acc **0.733854**（20,669/28,165）；方案 B Acc **0.735168**（20,706/28,165，Macro-F1 **0.717856**） |
| [实验三](./task3_fovea_localization/README.md) | 连通域 FOV 裁剪与逆映射 + `RFM`、`VisionFM`、`HRNet-W18` 编码器 + 共享多尺度空间解码头 + 末三层 LoRA（$r=\alpha=8$） + X/Y 分坐标轴 7 支路融合 | R30+V20 等权 80 图 OOF RMSE **17.36587 px**；7 支路最终融合 Y 轴 OOF RMSE **12.20757 px**（X 轴含全量 80 图模型，无独立 OOF） |
| [实验四](./task4_vessel_segmentation/README.md) | 图像级固定 5 折；全分辨率 `FR-UNet`（`FR96`）+ 原生分辨率先阈值再缩放置信度场 + `Pixel HGB`（71 维多尺度特征与冻结 `VascX` 先验）+ `ConvNeXt-LAB` + `QTSeg` + `RETFound-Green` 五路凸组合融合（阈值 0.30） | 20 图 OOF 宏平均 Dice **0.85259**（`FR96` 原生网格 **0.85130**） |

## 三、一键离线核验与重算

无需下载原始图像或模型权重即可校验全部四项实验的文件完整性、逐折/逐轮日志、归档指标、`submissions/` 下四份提交 CSV 的 SHA-256 指纹以及实验三的逐字节 CSV 重建：

```bash
python verify_all.py
```

各子项目独立核验入口：

- **实验一**：`cd task1_plant_seedling && python scripts/verify_results.py`
- **实验二**：`cd task2_facial_expression && python task2.py verify`（校验 `MANIFEST.json` 内全部 60 个受控文件 SHA-256；若已获取本地旧归档中的预测数组，可用 `python task2.py replay` 逐字节重放方案 A / 方案 B 提交文件）
- **实验三**：`cd task3_fovea_localization && python tools/rebuild_submission.py --verify`（从 7 个支路 `predictions/branches/*.csv` 重建并校验 `e5a8f5a...`；加 `--recipe configs/oof_candidate_fusion.json` 可重放 R30+V20 候选）
- **实验四**：`cd task4_vessel_segmentation && python scripts/verify_archive.py`

如需附带实验三官方训练坐标标签做额外复算：

```bash
python verify_all.py --task3-labels /path/to/fovea_localization_train_GT.csv
```

## 四、从头重新训练与数据/权重获取

各任务均提供完整的预训练权重获取、数据校验、训练与推理脚本，详细步骤请查阅各子目录 `README.md`：

1. 安装与本机 GPU/CUDA 驱动匹配的 PyTorch 与 torchvision（历史实验环境为 `torch 2.11.0+cu128`、`torchvision 0.26.0`），随后安装公共依赖：
   ```bash
   python -m pip install -r requirements.txt
   ```
2. **实验一**：支持直接命令行（`src/train_sprint.py` / `src/train_sprint_pseudo.py`）或 Kaggle Notebook（`notebooks/01_supervised_kaggle.ipynb`），加载固定修订版 `convnext_small.fb_in22k_ft_in1k` 与比赛 `2026task1` 数据运行 5 折训练与推理。
3. **实验二**：运行 `python fetch_assets.py --download` 获取并校验 5 个公开骨干权重及 DDAMFN++ 上游源码，运行 `python task2.py prepare --source <2026task2.zip>` 清洗分折后，用 `python task2.py train --scheme weighted` / `--scheme stacked` 及 `python task2.py fuse` 重训全部 7 组 35 折模型并输出提交 CSV。
4. **实验三**：运行 `python tools/prepare_weights.py --download` 校验并转换 RFM、VisionFM、HRNet 编码器，通过 `python tools/train_branch.py --branch <支路名>` 依次训练全部 7 个支路，再用 `tools/rebuild_submission.py` 按 `configs/fusion.json` 融合输出（`rfm_cv80` 与 `hrnet_rgb80` 保留原 Kaggle 环境检查，需在 Kaggle Notebook 中执行）。
5. **实验四**：运行 `python strong_vessels.py train`（依次使用 `configs/fr96.json`、`convnext.json`、`qtseg.json`、`retfound.json` 与 `--fold 0..4`）与 `python pixel_vessels.py` 重训五路分支，再用 `python scripts/reproduce.py teacher`、`predict`、`export --check-converter` 完成特征提取、融合推理与 RLE CSV 导出。

## 五、分发边界与开源许可

本仓库不包含原始课程竞赛图像/测试标注、大型预训练或训练后权重文件（`.pt` / `.pth` / `.safetensors`）、大型逐像素缓存及任何账号凭据。所引用的第三方模型代码与公开权重均遵循其原始开源许可证（见 [`task2_facial_expression/THIRD_PARTY_NOTICES.md`](./task2_facial_expression/THIRD_PARTY_NOTICES.md)、[`task2_facial_expression/licenses/`](./task2_facial_expression/licenses/) 与 `task4_vessel_segmentation/*_LICENSE.txt`）。由于 GPU 硬件、CUDA 与深度学习框架算子实现的浮点差异，从头重新训练得到的数值结果可能与历史归档存在细微波动；历史正式提交的逐字节一致性以各任务归档检查记录为准。
