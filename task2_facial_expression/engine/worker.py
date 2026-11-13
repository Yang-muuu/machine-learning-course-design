"""隔离各套历史源码的同名模块，调用其原始训练/预测函数。"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2 import ROOT, load as read, dump as write, sha, checked_weight as verify_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--mode', choices=['train', 'infer', 'smoke'], required=True)
    args = parser.parse_args()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # 必须在导入训练模块前设置隔离路径，否则不同版本的 CNN/Config 会互相污染。
    sys.path.insert(0, str(ROOT / 'engine' / 'vendor' / args.model))
    if args.model.startswith('ddam'):
        import fer_nets
        upstream = ROOT / 'local/ddam'
        for asset in read(ROOT/'configs/ddam_upstream.json').values():
            path = upstream / asset['adapter']
            if not path.is_file() or sha(path) != asset['adapter_sha256']:
                raise RuntimeError('请先运行 fetch_assets.py 准备上游 DDAM：' + str(path))
        fer_nets.__path__.append(str(upstream))
    import torch
    import numpy as np
    import pandas as pd
    from cnn_models import CNN, Config
    from cnn_train import fit_fold, predict, metrics
    from model_and_aug import EmotionDataset, make_loader, make_transform, seed_everything
    config = Config(**read(ROOT / 'configs' / f'{args.model}.json')).validate()
    device = torch.device(args.device)
    if device.type == 'cuda':
        assert torch.cuda.is_available(), '没有分配 GPU，请先运行 CPU smoke 或更换 GPU 运行时'
        # 使用真实设备运算发现 CUDA 驱动/安装问题，版本字符串不能代替此检查。
        print('GPU_READY', torch.cuda.get_device_name(device), float(torch.ones(1, device=device).sum()), flush=True)
    output = args.work / ('smoke_models' if args.mode == 'smoke' else 'models') / args.model
    weights, manifest = verify_weights(args.model)
    hashes = read(ROOT / 'configs/source_hashes.json')[args.model]
    context = {'name': args.model, 'cache': str(args.work / 'cache'),
        'split': str(args.work / 'reference_split.csv'), 'output': str(output),
        'weights': str(weights), 'device': args.device, 'deadline': time.time() + 7*24*3600,
        'source_hashes': hashes, 'pretrained_manifest': manifest}
    if args.mode == 'smoke':
        # 12 张真实训练图、6 张验证图、8 张测试图，一轮；不会混入正式五折结果。
        config = replace(config, epochs=1, max_epochs=1, min_epochs=1, warmup_epochs=0,
                         patience=0, batch_size=2, eval_batch_size=2, clean_train_samples=6,
                         workers=0, cpu_threads=1, unmixed_final_epochs=0)
        context['smoke'] = True
    output.mkdir(parents=True, exist_ok=True)
    write(output / 'context.json', context)
    write(output / 'environment.json', {'python': sys.version, 'torch_runtime': str(torch.__version__),
        'cuda_build': torch.version.cuda, 'device': args.device,
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'})
    if args.mode in ['train', 'smoke']:
        fit_fold(context, config, args.fold, 0)
    if args.mode == 'train':
        return
    # 独立从 best.pt 再推理，覆盖验证原有预处理、TTA、EMA 选中权重与 ID 顺序。
    run = output / 'runs' / f'fold_{args.fold}'
    saved = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    assert saved['fold'] == args.fold and bool(saved['smoke']) == (args.mode == 'smoke')
    assert saved['source_hashes'] == hashes
    assert json.loads(json.dumps(saved['config'])) == json.loads(json.dumps(config.dictionary()))
    seed_everything(config.seed + args.fold, config.cpu_threads)
    model = CNN(config, weights=None).to(device, memory_format=torch.channels_last)
    model.load_state_dict(saved['model'], strict=True)
    fixed = pd.read_csv(context['split'], dtype={'ID': str})
    rows = fixed.loc[fixed.validation_fold == args.fold].copy()
    if args.mode == 'smoke':
        rows = rows.groupby('label', group_keys=False).head(1)
    cache = Path(context['cache'])
    labels = pd.read_csv(cache / 'train_manifest.csv').label.to_numpy(dtype=np.int64)
    dataset = EmotionDataset(cache / 'train_images.npy', rows.array_index.to_numpy(), labels, make_transform(config, False))
    val = predict(model, make_loader(dataset, config, False, device, config.seed + args.fold), device)
    old_val = np.load(run / 'validation_probabilities.npy', allow_pickle=False)
    assert old_val.shape == val.shape
    max_difference = float(np.max(np.abs(old_val - val)))
    # CPU 与 GPU 浮点实现可能有微差；同一设备 smoke 要求严格对齐。
    tolerance = 1e-5 if args.mode == 'smoke' else 3e-3
    assert max_difference <= tolerance, max_difference
    test_count = 8 if args.mode == 'smoke' else len(pd.read_csv(cache / 'test_manifest.csv'))
    dataset = EmotionDataset(cache / 'test_images.npy', np.arange(test_count), None, make_transform(config, False))
    testing = predict(model, make_loader(dataset, config, False, device, config.seed + args.fold), device)
    test_diff = float(np.abs(testing - np.load(run / 'test_probabilities.npy', allow_pickle=False)).max())
    assert test_diff <= tolerance, test_diff
    # 保持正式训练产物不可被推理步骤覆盖；新推理概率单独导出。
    replay = run / 'inference_replay'
    replay.mkdir(exist_ok=True)
    np.save(replay / 'test_probabilities.npy', testing, allow_pickle=False)
    np.save(replay / 'validation_probabilities.npy', val, allow_pickle=False)
    rows.to_csv(replay / 'validation_rows.csv', index=False)
    report = {'model': args.model, 'fold': args.fold, 'smoke': args.mode == 'smoke',
        'checkpoint_sha256': sha(run / 'best.pt'), 'validation_max_difference': max_difference,
        'test_max_difference': test_diff, 'actual_validation_metrics': metrics(rows.label.to_numpy(int), val),
        'test_rows': len(testing), 'passed': True, 'device': args.device}
    write(replay / 'verification.json', report)
    print('INFERENCE_REPLAY_PASSED', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
