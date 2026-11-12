"""Run the recorded training entry point for one final fusion branch."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / 'configs/fusion.json').read_text(encoding='utf-8'))


def prepare(data, runtime):
    import pandas as pd

    runtime.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / 'training').glob('*.py'):
        shutil.copyfile(source, runtime / source.name)
    manifests = runtime / 'artifacts'
    subprocess.run([sys.executable, 'prepare_data.py', '--data', str(data), '--output', str(manifests)],
                   cwd=runtime, check=True)
    train_path = manifests / 'train.csv'
    train = pd.read_csv(train_path).sort_values('image_id')
    folds = pd.read_csv(ROOT / 'configs/fold_assignments.csv').sort_values('image_id')
    if train['image_id'].tolist() != folds['image_id'].tolist() or train['group'].tolist() != folds['group'].tolist():
        raise ValueError('Training image IDs or image hashes differ from the recorded experiment.')
    train['fold'] = folds['fold'].to_numpy()
    train.to_csv(train_path, index=False)


def training_commands(args, spec, output):
    command = [sys.executable, spec['script'], '--data', str(args.data),
               '--assets', str(args.assets), '--output', str(output)]
    if spec['folds'] == 1:
        return [command + ['--model', spec['model'], '--seed', str(spec['seed'])]]
    common = command + ['--worker', '--device', args.device]
    if spec.get('stage'):
        common += ['--stage', spec['stage']]
    elif spec.get('variant'):
        common += ['--variant', spec['variant'], '--seed', str(spec['seed'])]
    else:
        common += ['--model', spec['model'], '--seed', str(spec['seed']), '--epochs', str(spec['train_epochs'])]
    return [common + ['--fold', str(fold)] for fold in range(5)]


def collect_predictions(spec, output, branch_file):
    import pandas as pd

    if spec['folds'] == 1:
        source = pd.read_csv(output / f"epoch{spec['snapshot']}" / 'test_predictions.csv')
    else:
        pieces = []
        for fold in range(5):
            subdir = output / spec.get('variant', spec['model']) / f'fold{fold}'
            if spec.get('stage') or spec.get('variant'):
                subdir /= f"epoch{spec['snapshot']}"
            frame = pd.read_csv(subdir / 'test.csv')
            if frame['image_id'].nunique() != 20 or len(frame) != 20 or not frame['fold'].eq(fold).all():
                raise ValueError(f'Invalid predictions in {subdir}')
            pieces.append(frame)
        source = pd.concat(pieces).groupby('image_id', as_index=False)[['identity_x', 'identity_y']].mean()
    branch_file.parent.mkdir(parents=True, exist_ok=True)
    source[['image_id', 'identity_x', 'identity_y']].to_csv(branch_file, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/detection')
    parser.add_argument('--assets', type=Path, default=ROOT / 'weights/encoders')
    parser.add_argument('--work', type=Path, default=ROOT / 'work')
    parser.add_argument('--branch', choices=list(PLAN['branches']), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.data, args.assets, args.work = args.data.resolve(), args.assets.resolve(), args.work.resolve()
    spec = PLAN['branches'][args.branch]
    runtime = args.work / 'runtime' / args.branch
    output = args.work / 'runs' / args.branch
    commands = training_commands(args, spec, output)
    if args.dry_run:
        for command in commands:
            print(subprocess.list2cmdline(command))
        return
    if output.exists():
        raise FileExistsError(f'Use a new --work directory to preserve {output}.')
    if spec.get('kaggle_required') and not Path('/kaggle/working').is_dir():
        raise RuntimeError(f'{args.branch} uses its original Kaggle training entry point.')
    if spec['folds'] == 1 and args.device != 'cuda:0':
        raise ValueError('The two full-data refits use cuda:0.')
    output.mkdir(parents=True, exist_ok=False)
    prepare(args.data, runtime)
    for command in commands:
        subprocess.run(command, cwd=runtime, check=True)
    collect_predictions(spec, output, args.work / 'predictions' / f'{args.branch}.csv')


if __name__ == '__main__':
    main()
