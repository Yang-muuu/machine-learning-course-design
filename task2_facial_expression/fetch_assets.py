"""核验并准备公开初始化与上游 DDAM 源码；数据由使用者自行提供。"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
ALLOW = {
    'torch.nn.modules.linear.Identity', 'timm.models._efficientnet_blocks.SqueezeExcite',
    'timm.layers.adaptive_avgmax_pool.SelectAdaptivePool2d', 'timm.models._efficientnet_blocks.DepthwiseSeparableConv',
    'torch.nn.modules.activation.SiLU','timm.layers.norm_act.BatchNormAct2d',
    'torch.nn.modules.pooling.AdaptiveMaxPool2d','torch.nn.modules.conv.Conv2d',
    'timm.models._efficientnet_blocks.InvertedResidual','timm.models.efficientnet.EfficientNet',
    'torch.nn.modules.container.Sequential','torch.nn.modules.flatten.Flatten',
    'timm.layers.conv2d_same.Conv2dSame','torch.nn.modules.linear.Linear','torch.nn.modules.activation.Sigmoid',
}


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def checked(p, sha):
    if not p.is_file() or digest(p) != sha:
        raise ValueError(f'文件缺失或指纹不同：{p}')
    return p


def transfer(output, sha, *, url=None, source=None):
    if output.is_file():
        return checked(output,sha)
    output.parent.mkdir(parents=True,exist_ok=True)
    if source is None and url is None:
        raise FileNotFoundError(f'未准备：{output}；请提供 --from-engine 或 --download')
    with tempfile.NamedTemporaryFile(dir=output.parent,suffix='.partial',delete=False) as f:
        temp = Path(f.name)
        try:
            if source is not None:
                with checked(source,sha).open('rb') as upstream:
                    shutil.copyfileobj(upstream,f)
            else:
                req = Request(url, headers={'User-Agent':'Task2-course-reproduction'})
                with urlopen(req,timeout=90) as upstream:
                    shutil.copyfileobj(upstream,f)
            f.close()
            checked(temp,sha)
            temp.replace(output)
        finally:
            f.close()
            temp.unlink(missing_ok=True)
    print('已核验',output.name,flush=True)
    return output


def convert_b2(original, output, expected_sha):
    """仅在已核验的上游文件上允许指定 Torch/timm 类型，并严格重建。"""
    import torch
    import timm
    from safetensors.torch import save_file

    if timm.__version__ != '1.0.29':
        raise RuntimeError('转换需要 timm 1.0.29')
    names = set(torch.serialization.get_unsafe_globals_in_checkpoint(original))
    if not names <= ALLOW:
        raise ValueError(f'上游模型包含未允许的对象：{sorted(names-ALLOW)}')
    classes = [getattr(importlib.import_module(n.rsplit('.',1)[0]),n.rsplit('.',1)[1]) for n in sorted(names)]
    with torch.serialization.safe_globals(classes):
        saved = torch.load(original,map_location='cpu',weights_only=True).eval()
    rebuilt = timm.create_model('tf_efficientnet_b2',pretrained=False,num_classes=7,global_pool='max').eval()
    rebuilt.load_state_dict(saved.state_dict(),strict=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    temp = output.with_suffix('.partial')
    try:
        save_file({k:v.detach().cpu().contiguous() for k,v in rebuilt.state_dict().items()},str(temp))
        checked(temp,expected_sha)
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)


def prepare_ddam(meta, args):
    base = ROOT/'local/ddam'
    adapted = args.from_engine/'vendor/ddam_drop/fer_nets' if args.from_engine else None
    for item in meta.values():
        output = base/item['adapter']
        if output.is_file() or (adapted is not None and (adapted/item['adapter']).is_file()):
            transfer(output,item['adapter_sha256'],source=(adapted/item['adapter'] if adapted else None))
            continue
        original = transfer(base/'upstream'/item['filename'],item['sha256'],
                            url=item['url'] if args.download else None,
                            source=args.ddam_networks/item['filename'] if args.ddam_networks else None)
        value = original.read_bytes()
        if item['adapter']=='ddamfn_pp.py':
            old = b'from networks import MixedFeatureNet'
            if value.count(old) != 1:
                raise ValueError('DDAM 上游导入结构不符')
            value = value.replace(old,b'from . import mixed_feature_net as MixedFeatureNet')
        # 上游副本可能经 Git 自动换行。匹配历史适配字节后才写入。
        options = [value,value.replace(b'\r\n',b'\n'),value.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')]
        matched = next((b for b in options if hashlib.sha256(b).hexdigest()==item['adapter_sha256']),None)
        if matched is None:
            raise ValueError(f'DDAM 适配指纹不符：{item["filename"]}')
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_bytes(matched)
        print('已核验',output.name,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true',help='从来源清单中的官方地址下载')
    parser.add_argument('--from-engine',type=Path,help='已有离线复现包内的 engine 目录')
    parser.add_argument('--ddam-networks',type=Path,help='已取得的指定 DDAMFN++ revision 下的 networks 目录')
    parser.add_argument('--no-weights',action='store_true',help='仅准备或校验 DDAM 源码')
    args=parser.parse_args()
    ddam = json.loads((ROOT/'configs/ddam_upstream.json').read_text(encoding='utf-8'))
    prepare_ddam(ddam,args)
    if args.no_weights:
        return
    records = json.loads((ROOT/'configs/pretrained.json').read_text(encoding='utf-8'))
    seen = set()
    for key, item in records.items():
        name = item['filename']
        if name in seen:
            continue
        seen.add(name)
        output = ROOT/'local/weights'/name
        if key=='b2_native' and args.download and not args.from_engine and not output.is_file():
            original = transfer(ROOT/'local/upstream/enet_b2_7.pt',item['original_sha256'],url=item['original_url'])
            convert_b2(original,output,item['sha256'])
        else:
            transfer(output,item['sha256'],url=item.get('url') if args.download else None,
                     source=args.from_engine/'weights'/name if args.from_engine else None)
        if output.stat().st_size != item['bytes']:
            raise ValueError(f'权重大小不符：{output}')
    print('公开初始化及上游依赖已准备；未下载比赛数据。',flush=True)


if __name__ == '__main__':
    main()
