# 机器学习课程设计实验二：人脸表情分类

将 48×48 灰度人脸分为六类，按比赛模板生成 `ID,Emotion`。类别顺序为愤怒、恐惧、快乐、悲伤、惊讶、中性（0–5）。

本项目实现最终的两种方案：

| 方案 | 训练与融合 | 训练侧准确率 | Kaggle 成绩（提交者记录） |
| --- | --- | ---: | ---: |
| A `weighted` | ConvNeXt-Tiny＋ECA、DINOv2-S/14、DDAMFN++（Dropout 0.30）、DAN；五折概率按 0.20 / 0.20 / 0.40 / 0.20 融合 | 20,669 / 28,165 = 0.733854 | 0.747 |
| B `stacked` | A 的四个配置，加 EmotiEffNet-B2、DDAMFN++（Dropout 0.20）和 DDAMFN++（SAM）；42 维对数概率、标准化和带 L2 正则的多分类逻辑回归 | 20,706 / 28,165 = 0.735168 | 0.747 |

A 的数值是固定配方的完整五折训练侧 OOF；B 的数值是固定基础模型 OOF 上对融合层再做的五折交叉拟合，包含融合参数选择。两者都不是另行留出的独立验证集。B 历史选用 `C=0.01`。每个基础模型按同一份像素分组划分训练，验证图由留出该组的模型预测；测试图在每折做原图/水平翻转预测，再按折和模型融合。

## 文件

- `task2.py`：数据准备、划分验证、训练、四模型融合、七模型融合、历史概率重放、预测 CSV 比较。
- `engine/vendor/`：七套实际运行的主要模型、增强和折训练源码；`engine/worker.py` 隔离不同版本的同名模块。
- `configs/`：冻结超参数、原验证折号、公开初始化来源及 SHA256。折号文件按原缓存行顺序保存，不含图像、文件 ID 或标签；准备数据时重新对齐验证。
- `evidence/`：训练侧汇总、每折训练曲线、配对结果与最终两个 CSV 的指纹；没有逐图预测文件。
- `fetch_assets.py`：可选下载和验证公开预训练文件，或从现有离线归档导入相同初始化；DDAMFN++ 的上游模型源码留在忽略目录 `local/`，见 `THIRD_PARTY_NOTICES.md`。

## 运行环境

Python 3.11–3.13。训练需要与驱动配套的 PyTorch 和 torchvision，优先沿用 GPU 平台已安装的版本。准备数据、核验或重放只需 NumPy、pandas、Pillow；重新拟合融合层需要 scikit-learn。安装其余依赖：

```bash
python -m pip install -r requirements.txt
python task2.py verify
```

## 从原图训练并生成预测

把课程数据 ZIP 放到 `local/2026task2.zip`，或为 `--source` 指定 ZIP / 已解压的数据目录。先准备公开初始化和 DDAMFN++ 上游源码：

```bash
python fetch_assets.py --download
```

已有之前的本地完整归档时，也可以直接导入该归档 `engine/` 下的公开初始化和 DDAM 适配源码：

```bash
python fetch_assets.py --from-engine /path/to/local_archive/engine
```

下载文件的版本和指纹见 `configs/pretrained.json`、`configs/ddam_upstream.json`。如果外部站点无法连通，可按清单自行取得文件，再运行核验或离线导入。下载 B2 原始 `.pt` 后，脚本用固定的可加载对象清单转成纯张量 `safetensors`。

```bash
python task2.py prepare --source local/2026task2.zip
python task2.py smoke --model convnext_eca --device cpu

# 四模型共 20 折，完成后生成方案 A
python task2.py train --scheme weighted --device cuda:0
python task2.py fuse --scheme weighted --output work/final_weighted

# 共用已经完成的四组模型，仅补训练新增的三组五折
python task2.py train --scheme stacked --device cuda:0
python task2.py fuse --scheme stacked --output work/final_stacked
```

冒烟训练只用 12 张真实训练图、6 张验证图和 8 张测试图运行一轮并验证重载；其结果保存在 `work/smoke_models/`，不进入正式评分。可用 `--model ddam_drop --folds 0 1` 分批训练；完整折复核后跳过，未完成折保留备份后重跑。融合要求所有对应五折均完整，并核对验证样本的 ID、标签、像素指纹与预测顺序。每个正式输出含 CSV、融合概率和训练侧指标。重新训练可能因 CUDA、混合精度和算子差异产生不同预测，不能保证再次取得相同分数。

## 核验原提交与精确重放

把自己的两份提交传给比较入口；该命令按 ID 对齐并统计预测差异，不会把差异解释成错分。

```bash
python task2.py compare-submissions --first "submission (5).csv" --second "submission (6).csv"
```

精确重放需要先有旧归档中的真实冻结预测与对应模板；堆叠另需旧的 `stacking_parameters.npz`。用旧的七模型归档时，两套方案共用 `predictions/` 和 `data/sample_submission.csv`：

```bash
python task2.py replay --scheme weighted --predictions /path/to/archive/predictions --sample /path/to/archive/data/sample_submission.csv --output work/replay_a
python task2.py replay --scheme stacked --predictions /path/to/archive/predictions --sample /path/to/archive/data/sample_submission.csv --parameters /path/to/archive/stacking_parameters.npz --output work/replay_b
```

历史重放核对两个最终 CSV 的原始 SHA256，属于融合层复现。公开项目不包含比赛图像、历史逐样本预测、训练权重或账号信息，也不会调用比赛提交接口。
