# 实验一 植物幼苗图像分类

用 500 张有标签的 RGB 图训练五分类模型，每类 100 张。正式方案是 ConvNeXt-Small 的分层五折监督训练。每折按验证集 Macro-F1 选权重，推理时平均五折的类别概率。训练集折外预测共判对 485 张，准确率 0.9700，Macro-F1 0.97010。

| 路径 | 用途 |
| --- | --- |
| `notebooks/01_supervised_kaggle.ipynb` | 监督训练与推理的 Kaggle 入口，所需 Python 源码已内嵌。 |
| `src/train_sprint.py` | 监督训练与推理的命令行入口。 |
| `src/sprint_refined.py` | 五折训练、输入增强、损失、EMA 和选模。 |
| `src/original_sprint.py` | 历史源码，供对照实现。 |
| `results/` | 训练集折外预测、逐轮日志、指标摘要和曲线。 |
| `scripts/verify_results.py` | 用 Python 标准库复算归档指标，校验日志与可选预测 CSV。 |

## 监督实验

在 Kaggle Notebook 中挂载课程比赛 `2026task1` 的官方数据并选用 16 GB Tesla T4。启用网络下载固定的 ImageNet 权重，然后按单元格顺序运行监督 Notebook。程序先用 100 张训练图、20 张留出的训练图和 8 张推理图做一轮冒烟检查，正式五折配置为 448 × 448、batch size 12、40 轮和种子 42。每次运行把预测保存为 `/kaggle/working/plant_runs/<运行编号>/submission.csv`。

本地运行前准备与 GPU、驱动匹配的 PyTorch 和 torchvision，并安装 `requirements.txt`。历史正式实验使用 PyTorch `2.11.0+cu128` 和 timm `1.0.29`。从 [timm 权重仓库](https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k/tree/fe6b358aa409414cdb522ae3d43a2c9922f7cc1b) 获取 `model.safetensors`。数据根目录包含一对同级 `train/` 与 `test/`，以及官方 `submission-for-task2.csv`。可保留压缩包原有的嵌套层级。

```bash
python -m pip install -r requirements.txt
python src/train_sprint.py --data-root data/2026task1 --out-dir outputs/smoke --weights weights/model.safetensors --smoke --smoke-batch 4 --max-minutes 8
python src/train_sprint.py --data-root data/2026task1 --out-dir outputs/supervised --weights weights/model.safetensors --max-minutes 110
```

输出目录请用新目录。正式训练完成后在 `outputs/supervised/submission.csv` 生成按模板 ID 排序的 378 行预测。冒烟用于核对训练与推理流程。归档数值来自历史运行，设备和库版本改变时可能出现数值差别。

## 对照实验和榜单记录

历史上另做了一次无标签测试图自训练。五折教师对 378 张图推理，按平均最大概率至少 0.90、五个教师类别一致和每类最多 50 张的条件得到 208 张软伪标签。五个学生分别从对应教师初始化，各用全部 500 张有标签训练图和相同的 208 张伪标签图继续训练五轮。伪标签损失权重为 0.20。教师与学生等权融合，最终类别与纯监督预测按 ID 比较一致。学生没有独立的折外结果，这次比较也没有把增加有标签样本与伪标签两个因素拆开。

Kaggle 历史提交记录 57043322 对应这份教师与学生融合 CSV，公开榜单 `MeanF1Score` 为 0.9708。归档保存该 CSV 的 SHA-256，历史监督 CSV 与它字节不同，但 378 个 ID 对应的类别一致。正式方法采用纯监督，便于复查训练数据范围。只有课程规则允许利用无标签测试图训练时，才启用 `notebooks/02_pseudolabel_comparison_kaggle.ipynb` 中的开关或运行 `src/train_sprint_pseudo.py --allow-unlabeled-test-training`。

## 离线核对

```bash
python scripts/verify_results.py
python scripts/verify_results.py --submission outputs/supervised/submission.csv --template data/2026task1/submission-for-task2.csv
```

第一个命令从 500 条保存的折外预测复算准确率、宏平均 F1 和混淆矩阵，并核对监督阶段 200 条、学生阶段 25 条轮次日志。第二个命令额外验证 CSV 格式、类别、ID 和顺序，并输出字节及按 ID 对齐后的历史比对结果。若专门核对历史归档，可加 `--expect-historical-mapping`。若手头有两份预测，则用 `--compare` 报出相异预测数量。榜单记录可在 `results/summary.json` 查到。重新训练需要自行获取官方数据和权重。
