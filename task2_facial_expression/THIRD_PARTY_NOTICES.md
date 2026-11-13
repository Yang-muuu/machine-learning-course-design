# 上游来源

- ConvNeXt-Tiny：[Liu 等，A ConvNet for the 2020s](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html)。骨干由 `torchvision` 提供，ECA 按 [Wang 等](https://arxiv.org/abs/1910.03151) 的方法在最后卷积特征图上实现。
- DINOv2：[Oquab 等](https://arxiv.org/abs/2304.07193)，通过 `timm` 使用 ViT-S/14 和固定版本的 safetensors 初始化。
- DDAMFN++：[SainingZhang/DDAMFN](https://github.com/SainingZhang/DDAMFN)，模型接口适配和训练代码在本项目中，上游两个模型文件由 `fetch_assets.py` 按版本和指纹准备在 `local/ddam/`，不纳入本仓库。
- DAN：[Wen 等](https://arxiv.org/abs/2109.07270) 及 [yaoing/DAN](https://github.com/yaoing/DAN)。DAN 上游文件保留 MIT 许可，见 `licenses/DAN_LICENSE`。
- EmotiEffNet-B2：[sb-ai-lab/EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib)。用 `timm` 重建原生 260 输入和最大池化，公开权重来源见清单。

比赛数据和上游预训练权重由使用者从对应来源取得，适用其发布者条款。`local/` 用于保存外部依赖，默认排除在版本控制之外。
