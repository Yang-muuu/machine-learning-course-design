# 任务四 眼底血管分割

报告中的 `submission (11).csv` 对应提交 **57947747**，Kaggle 公榜 Dice 为 **0.86267**，私榜为 **0.86530**。提交文件的 SHA-256 与 `records/final_snapshot.json` 一致。

固定五折按图像划分，每折用 16 张图训练、4 张图验证。所有模型共用划分。每张图由未用该图拟合参数的折模型预测，再计算 512×512 全图的血管 Dice。五路融合在这 20 张图上的宏平均 Dice 为 **0.85259**。模型、融合权重和阈值都参考了这批折外（OOF）结果，分数用于比较开发方案。

| 方案 | 阈值 | OOF Dice |
| --- | ---: | ---: |
| 固定 FR96，先双线性缩放概率，再阈值化 | 0.50 | 0.76507 |
| 固定 FR96，先在原图网格阈值化，再按附件转换规则栅格化 | 0.50 | 0.85127 |
| 同一 FR96 栅格化方案，按 OOF 网格选择阈值 | 0.475 | 0.85130 |
| FR96、Pixel HGB、ConvNeXt、QTSeg 与 RETFound-Green 融合 | 0.30 | 0.85259 |

FR96 对照使用同一套权重、视野掩码和本地标签。转换规则是将 8 位灰度掩码用 Pillow 双线性缩放至 512×512，取非零像素为血管。融合的有效权重依次为 0.458145、0.295245、0.100000、0.081000、0.065610，最终预测不做额外形态学操作。各成员在固定阈值和网格择优时的复算结果见 `records/member_comparison.json`。

## 复查归档结果

在本任务目录运行下列命令。此步骤只需 Python 3.11 及以上版本。

```powershell
python -B scripts/verify_archive.py
```

脚本用 `records/oof_counts.json` 中每张训练图的 TP、FP、FN 复算 Dice、精确率、召回率和 IoU。它也核对五折划分、融合权重和四路神经网络日志中的 2050 条训练轮次记录。`records/final_snapshot.json` 保存各折入选权重的 SHA-256；这一步验证归档内的记录，不执行模型训练或推理。

## 用原始附件和已存分数图复核

从课程 [2026task-4](https://www.kaggle.com/competitions/2026task-4) 取得原始附件，保持以下布局。用 `--data` 指定解压目录。

```text
data/segmentation/
  train/image/
  train/label/
  test/image/
  sample_submission.csv
  segmentation_to_csv.py
```

复核分数图需要 NumPy、SciPy 和 Pillow，版本见 `requirements.txt`。使用 `--check-converter` 还需 pandas 和与 PyTorch 匹配的 torchvision，供原始附件的转换脚本使用。若保留了原实验的 `delivery/goal_085_20261103`，将示例中的 `snapshot` 替换为该目录。

```powershell
python scripts/reproduce.py score-oof --data data/segmentation --predictions snapshot/oof_predictions.npz
python scripts/reproduce.py export --data data/segmentation --cache snapshot/member_predictions --out runs/replay --check-converter --exact
```

第一条命令检查原始训练掩码的摘要，重做本地标签转换，并逐图比较像素计数和 Dice。第二条命令按冻结顺序重放五路 float32 混合，导出 20 张二值 PNG 和 `submission.csv`。PNG 的黑色代表血管，CSV 对血管按行优先、从 1 开始做游程编码。对照附件转换脚本时会把用于核验的 PNG 反色，使其非零像素表示同一血管区域。`--exact` 要求输出 CSV 的 SHA-256 为 `4d516ae13d88d356dfe5ecc9a17f4f6609b1383c469c7de92b6e809ef2bf5d32`。这是文件重放检查。

## 从权重推理或重训

推理和训练需要与硬件匹配的 PyTorch、torchvision 以及本目录的依赖。原实验配置参考 PyTorch 2.11.0、torchvision 0.26.0。训练建议使用 CUDA。权重和数据由使用者自行取得，公开初始化记录见 `records/pretraining/` 和 `configs/pixel.json`。

已保存的四路神经网络权重应放在 `--model-dir` 下，路径为 `<weights_folder>/results/fold_N/best.pt`，其中 N 为 0 至 4，目录名见 `records/final_snapshot.json`。Pixel HGB 权重在 `--pixel-dir` 下，包含 `fold_N/model.joblib` 和 `fold_N/split.json`。只加载来源可信的权重。

```powershell
python scripts/reproduce.py teacher --data data/segmentation --split test --out runs/teacher_test.npz
python scripts/reproduce.py predict --data data/segmentation --split test --model-dir optimization/best_weights --pixel-dir optimization/local_pixel_hgb_v1/results --teacher runs/teacher_test.npz --out runs/members
python scripts/reproduce.py export --data data/segmentation --cache runs/members --out runs/export
```

用原实验权重时可为 `predict` 加 `--exact-weights`。用原训练图复核折外推理时，先生成 `--split train` 的 VascX 特征，再用 `predict --split oof`，最后以 `score-oof --cache` 对融合结果评分。VascX 分数来自冻结的公开血管模型，在 HGB 中作为输入特征。HGB 以各折训练图的标注拟合。

重训时，四路神经网络分别使用下面的配置和输出目录。示例中的折号从 0 改到 4，输出目录使用新的运行路径。FR96、QTSeg 和 RETFound 采用均匀随机裁块及旋转、翻转增强，ConvNeXt 使用随机仿射采样。训练时保留原有检查点选择口径，冻结后再按附件转换口径复核。不同硬件和软件版本可能产生不同权重。

| 配置 | 权重目录 |
| --- | --- |
| `configs/fr96.json` | `frunet_native96_v8` |
| `configs/convnext.json` | `convnext_lab_v1` |
| `configs/qtseg.json` | `qtseg_patch512_v4` |
| `configs/retfound.json` | `retfound_native_v11` |

```powershell
python strong_vessels.py train --config configs/fr96.json --data data/segmentation --fold 0 --device cuda:0 --out runs/models/frunet_native96_v8/results
python scripts/reproduce.py teacher --data data/segmentation --split train --out runs/teacher_train/reference_predictions.npz
python pixel_vessels.py --data data/segmentation --teacher runs/teacher_train/reference_predictions.npz --out runs/pixel
```

归档包含源码、配置和小型验证记录。课程图像与标注、训练权重、逐像素缓存和最终预测文件单独保管。上游代码和权重按各自许可证使用；许可和源码出处见 `*_LICENSE.txt`、`rtnls_fundusprep/UPSTREAM.json` 以及对应权重记录。
