# 任务二 人脸表情分类

## Kaggle 最终提交

报告中的 `submission (9).csv` 对应提交 **57273426**，公榜与私榜 Accuracy 均为 **0.747**。四路分别为 ConvNeXt-Tiny + ECA、DINOv2 ViT-S/14、DDAMFN++、DAN。每路平均五折模型输出的六类概率，再按 0.20、0.20、0.40、0.20 融合。训练侧四路融合的完整五折准确率为 73.39%，Macro-F1 为 0.71595。

[最终配方](records/kaggle_final_recipe.json)保存四路训练配置、选用轮次和预测归档指纹。开发阶段曾用测试参考核对冻结预测，模型分析采用分组折外结果。使用原始项目的 `exact_replay/frozen_predictions.zip` 可重建提交 CSV。回放入口使用 NumPy，读取四路概率与模板 ID，并校验输出 SHA-256。

~~~bash
python tools/replay_kaggle_final.py --archive /path/to/frozen_predictions.zip
~~~

## 训练侧 ResNet 对照

本实验对 48 × 48 灰度人脸图像做六分类。标签顺序为愤怒（Angry，0）、恐惧（Fear，1）、高兴（Happy，2）、悲伤（Sad，3）、惊讶（Surprise，4）、中性（Neutral，5）。从课程原始训练集隔离 39 个标签冲突的同像素组，共 110 张，剩余 28,165 张。其余同像素图保留，用解码灰度像素的 SHA-256 分组，生成固定五折划分。本次训练使用第 0、1、2 折验证，覆盖 16,898 张。第 0 折用于筛选 ResNet34 的学习率，第 1、2 折复查选定配置。

两种骨干均从 torchvision 的 ImageNet 权重初始化，将灰度图复制为三通道后缩放至 96 × 96。保留原始残差结构和全局平均池化，用 Dropout 与六类线性层替换分类头，并微调整网。训练时做轻量仿射、翻转和亮度对比度扰动。验证时平均原图与水平翻转图的概率，按验证准确率选 checkpoint，同分时取较低 Log-loss。

| 方法 | 第 0 折 | 第 1 折 | 第 2 折 | 三折合并准确率 | Macro-F1 | Log-loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 68.47% | 67.79% | 67.04% | 67.77% | 0.6591 | 1.0005 |
| ResNet34 | 68.73% | 68.04% | 66.83% | 67.87% | 0.6600 | 1.0397 |
| 概率等权融合 | 69.95% | 69.14% | 67.96% | 69.01% | 0.6717 | 0.9309 |

等权融合在三个已做的折中分别好于两种单模型，三折合并比 ResNet18 多判对 211 张。ResNet34 与 ResNet18 在第 1、2 折的平均准确率只差约 0.02 个百分点。两种骨干采用不同学习率，且每折只使用一个训练随机种子，不能据此单独判断网络深度的作用。

课程数据中训练与无标签测试图像存在 482 个相同的灰度像素哈希。本次结果来自训练集内按像素哈希分组的验证。代码的测试导出只读取图像及模板 ID 顺序，类别来自模型概率的最大值。记录中没有本次方案的榜单成绩。精确像素分组不保证近重复或同一人物完全隔离。

## 查看与复核归档

使用 Python 3.10 或更新版本，从本目录运行。此命令不需要下载数据、模型或依赖。

```bash
python -B verify_results.py
```

脚本核对训练代码指纹和五折分组计数，从三个已评估折的保存概率复算两种单模型和等权融合的准确率、混淆矩阵、Macro-F1、Log-loss 和各类召回率，并从 175 条训练轮次记录核对最佳轮次。归档中的分组 bootstrap 区间来自原实验，不在该命令中重算。该命令只复核归档数据与计算过程。

- `records/data_audit.json` 保留分组与清洗计数、课程原始 ZIP 和固定划分的 SHA-256。
- `records/split_groups.csv.gz` 记录固定五折各匿名组的类别、样本数及验证折。
- `records/fold_*_validation.csv.gz` 包含训练集留出折的匿名组、训练标注和两模型验证概率。归档中的标签只对应训练数据。
- `records/history/` 收录六次正式训练与一次学习率对照的逐轮日志。`records/summary.json` 记录历史指标和原始验证表的指纹。

## 使用课程数据重新训练

需要老师提供的原始 `2026task2.zip` 或保留原目录结构的解压目录，其中应包含 `train/<类别>/*.jpg`、`test/*.jpg` 和唯一的 `ID,Emotion` 提交模板。历史环境为 Python 3.13.7、PyTorch 2.11.0、torchvision 0.26.0、CUDA 12.8 与 RTX 4060 Laptop GPU。请先安装适合本机的 PyTorch 和 torchvision，再安装其余依赖。预训练权重由 torchvision 官方加载，首次运行需要下载或事先缓存。

```bash
python -m pip install -r requirements.txt
python -X utf8 train_emotion.py audit --data /path/to/2026task2.zip
python -X utf8 train_emotion.py train --data /path/to/2026task2.zip --device cuda --arch resnet18 --n-splits 5 --folds 0,1,2 --seed 42 --epochs 25 --patience 0 --lr 0.0005 --label-smoothing 0.1 --validation-only --output runs/resnet18
python -X utf8 train_emotion.py train --data /path/to/2026task2.zip --device cuda --arch resnet34 --n-splits 5 --folds 0,1,2 --seed 42 --epochs 25 --patience 0 --lr 0.0003 --label-smoothing 0.1 --validation-only --output runs/resnet34
python -B verify_results.py --split-manifest runs/resnet34/split_manifest.csv
python -B verify_results.py --runs runs/resnet18 runs/resnet34
```

两次正式训练的共同参数包括 batch size 128、Dropout 0.2、AdamW 权重衰减 1e-4、分类头学习率为骨干三倍、线性预热两轮后余弦衰减、每折随机种子为 42 加折号。若复跑第 0 折的 ResNet34 学习率筛选，可在相同设置下另跑 `--folds 0 --lr 0.0005 --output runs/resnet34_lr5e4`，与 0.0003 的对应折结果比较。历史规则按准确率、Macro-F1、较低 Log-loss 的顺序选择配置。ResNet18 使用早期参照的 0.0005 配置，以上共用训练入口保留其训练参数。硬件、库版本和随机数实现可能影响重新训练的数值。

## 导出测试预测

用相同折数分别训练两个模型后，可将六个 checkpoint 的预测概率直接取平均。这样两种模型各占一半。示例生成本地 CSV，行序与课程模板一致。
使用历史六份 checkpoint 和课程原始数据重放后，生成的 7,065 行 CSV 与原等权融合导出逐字节一致。`records/export_check.json` 保留输出指纹与格式检查记录。CSV 和 checkpoint 由本机运行时产生。

```bash
python -X utf8 train_emotion.py predict --data /path/to/2026task2.zip --device cuda --output runs/prediction --checkpoints runs/resnet18/fold_0/best.pt runs/resnet18/fold_1/best.pt runs/resnet18/fold_2/best.pt runs/resnet34/fold_0/best.pt runs/resnet34/fold_1/best.pt runs/resnet34/fold_2/best.pt
```

骨干出处为 He 等人的 [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)。训练通过安装的 torchvision 调用模型定义，仓库没有附带 torchvision 源码或模型权重。
