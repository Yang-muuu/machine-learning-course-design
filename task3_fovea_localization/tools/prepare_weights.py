"""Prepare the three pinned pretrained encoders from their public source checkpoints."""

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
MANIFEST = json.loads((ROOT / 'configs/pretrained_sources.json').read_text(encoding='utf-8'))


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(spec, source):
    if source.exists():
        return
    if spec.get('google_drive_id'):
        import gdown

        if gdown.download(id=spec['google_drive_id'], output=str(source), quiet=False) is None:
            raise RuntimeError(f'Download failed: {spec["url"]}')
    else:
        with urllib.request.urlopen(spec['url'], timeout=120) as request, source.open('wb') as output:
            shutil.copyfileobj(request, output)


def convert(name, source, destination):
    if name == 'hrnet':
        shutil.copyfile(source, destination)
        return
    import numpy as np
    import torch
    from next_models import backbone
    from safetensors.torch import save_file

    safe_types = [(np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'), np.dtype,
                  type(np.dtype('float64')), type(np.dtype('float32')), argparse.Namespace]
    with torch.serialization.safe_globals(safe_types):
        checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    if name == 'rfm':
        state = {key.removeprefix('backbone.'): value
                 for key, value in checkpoint['model_state_dict'].items()}
    else:
        state = {key.removeprefix('backbone.'): value
                 for key, value in checkpoint['teacher'].items() if key.startswith('backbone.')}
    model, _ = backbone(name, input_size=224)
    model.load_state_dict(state, strict=True)
    save_file({key: value.detach().cpu().contiguous() for key, value in state.items()}, str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'weights/raw')
    parser.add_argument('--assets', type=Path, default=ROOT / 'weights/encoders')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    args.assets.mkdir(parents=True, exist_ok=True)
    for name, spec in MANIFEST.items():
        output = args.assets / spec['filename']
        if output.exists() and sha256(output) == spec['sha256']:
            print(f'{name} verified')
            continue
        source = args.raw / spec.get('original_filename', spec['filename'])
        if args.download:
            fetch(spec, source)
        if not source.exists():
            raise FileNotFoundError(f'Place {spec["url"]} at {source}, or use --download.')
        if sha256(source) != spec.get('original_sha256', spec['sha256']):
            raise ValueError(f'Source checkpoint checksum mismatch: {source}')
        temporary = output.with_suffix('.staged.safetensors')
        convert(name, source, temporary)
        if sha256(temporary) != spec['sha256']:
            raise ValueError(f'Converted encoder checksum mismatch: {temporary}')
        temporary.replace(output)
        print(f'{name} verified')
    (args.assets / 'source_provenance.json').write_text(json.dumps(MANIFEST, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
