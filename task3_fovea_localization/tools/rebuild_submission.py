"""Rebuild a Task 3 CSV from recorded branches and coordinate weights."""

import argparse
import csv
import hashlib
import io
import json
import re
from decimal import Decimal, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / 'configs/fusion.json').read_text(encoding='utf-8'))


def read_branch(path):
    result = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            image_id = int(row['image_id'])
            point = (Decimal(row['identity_x']), Decimal(row['identity_y']))
            if image_id in result or not all(v.is_finite() and v >= 0 for v in point):
                raise ValueError(f'Invalid prediction for {image_id} in {path}')
            result[image_id] = point
    return result


def build(branch_dir, sample, plan=PLAN):
    weights = {axis: {name: Decimal(value) for name, value in items.items()}
               for axis, items in plan['weights'].items()}
    if set(weights) != {'X', 'Y'}:
        raise ValueError('Expected weights for X and Y.')
    used = set(weights['X']) | set(weights['Y'])
    if not all(re.fullmatch(r'[a-z0-9_]+', name) and name in plan['branches'] for name in used):
        raise ValueError('Unexpected branch in recipe.')
    for axis_weights in weights.values():
        if any(not w.is_finite() or w <= 0 for w in axis_weights.values()) or sum(axis_weights.values()) != 1:
            raise ValueError('The weights on each axis must be positive and sum to one.')
    tables = {name: read_branch(branch_dir / f'{name}.csv') for name in sorted(used)}
    with sample.open(encoding='utf-8-sig', newline='') as stream:
        order = [row['ImageID'] for row in csv.DictReader(stream)]
    expected = {f'{n}_Fovea_{axis}' for n in range(81, 101) for axis in 'XY'}
    if len(order) != 40 or set(order) != expected:
        raise ValueError('Expected one X row and one Y row for images 81 through 100.')
    if any(set(table) != set(range(81, 101)) for table in tables.values()):
        raise ValueError('Each branch must provide exactly the 20 test images.')
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\r\n')
    writer.writerow(['ImageID', 'value'])
    with localcontext() as context:
        context.prec = 50
        for key in order:
            match = re.fullmatch(r'(\d+)_Fovea_([XY])', key)
            image_id, axis = int(match[1]), match[2]
            index = 'XY'.index(axis)
            value = sum(coef * tables[name][image_id][index]
                        for name, coef in weights[axis].items())
            writer.writerow([key, f'{value:.8f}'])
    return output.getvalue().encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, default=ROOT / 'configs/fusion.json',
                        help='coordinate blend recipe; the default is the training OOF candidate')
    parser.add_argument('--branch-dir', type=Path, default=ROOT / 'predictions/branches')
    parser.add_argument('--sample', type=Path, default=ROOT / 'configs/submission_order.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'work/submission.csv')
    parser.add_argument('--verify', action='store_true', help='check the archived submission checksum')
    args = parser.parse_args()
    plan = json.loads(args.recipe.read_text(encoding='utf-8'))
    payload = build(args.branch_dir, args.sample, plan)
    checksum = hashlib.sha256(payload).hexdigest()
    if args.verify and checksum != plan['submission_sha256']:
        raise ValueError(f'Submission checksum mismatch: {checksum}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f'{args.output}\n40 coordinate rows\nSHA256 {checksum}')


if __name__ == '__main__':
    main()
