"""Train five supervised folds and export their average predictions."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sprint_refined as sprint

CLASSES = ['Black-grass', 'Common wheat', 'Loose Silky-bent', 'Scentless Mayweed', 'Sugar beet']
ARCH = 'convnext_small.fb_in22k_ft_in1k'


def probability_check(prob, count):
    if prob.shape != (count, 5) or not np.isfinite(prob).all() or (prob < 0).any():
        raise ValueError('Invalid predictions')
    if not np.allclose(prob.sum(1), 1, atol=.002):
        raise ValueError('Probabilities do not sum to one')


def data_index(root):
    return sprint.read_course_data(root, CLASSES)


def write_csv(path, prob, testing, template):
    probability_check(prob, 378)
    predictions = {Path(item[0]).name: CLASSES[int(prob[i].argmax())]
                   for i, item in enumerate(testing)}
    if set(predictions) != set(template):
        raise ValueError('Submission IDs do not match')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ID', 'Category'])
        writer.writerows((image_id, predictions[image_id]) for image_id in template)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--submission-path', help='Defaults to submission.csv inside the run directory')
    parser.add_argument('--max-minutes', type=float, default=110.)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-batch', type=int, choices=[4, 12], default=4)
    args = parser.parse_args()
    if not 1 <= args.max_minutes <= 112:
        parser.error('Invalid runtime budget')
    started = time.monotonic()
    deadline = started + args.max_minutes * 60
    folder = Path(args.out_dir).absolute()
    folder.mkdir(parents=True, exist_ok=True)
    if not args.smoke and any(folder.iterdir()):
        raise RuntimeError('Use an empty output folder for a complete run')
    submission = Path(args.submission_path).absolute() if args.submission_path else folder / 'submission.csv'
    if not args.smoke:
        if submission.exists():
            raise FileExistsError(f'Choose a new submission path: {submission}')
    weights = Path(args.weights).absolute()  # retain .safetensors on Linux symlinks
    if not weights.is_file():
        raise FileNotFoundError('Approved ImageNet pretrained weights are required')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('CUDA is required for this training workflow')
    import timm
    train, test, training, testing, template = data_index(args.data_root)

    def check_budget(_model=None, _input=None):
        if time.monotonic() >= deadline:
            raise TimeoutError('Complete training exceeded its runtime budget')

    def build_model(cfg, count):
        if cfg['arch'] != ARCH:
            raise ValueError('Keep the reported .97 ConvNeXt-Small architecture')
        model = timm.create_model(ARCH, pretrained=True, num_classes=count,
                                  drop_rate=cfg['drop_rate'], drop_path_rate=cfg['drop_path_rate'],
                                  pretrained_cfg_overlay={'file': str(weights)})
        model.register_forward_pre_hook(check_budget)
        return model

    sprint.build_model = build_model
    sprint.find_dirs = lambda *a, **k: (str(train), str(test))
    # Keep the complete source's hyperparameters, DataLoaders and selection rule.
    sprint.CFG['out_dir'] = str(folder)
    cfg = dict(sprint.CFG)
    report = dict(smoke=args.smoke,
                  original_recipe_preserved=not args.smoke, input_mode='original RGB',
                  workers=cfg['num_workers'], resolution=448, architecture=ARCH,
                  teacher_labels_only=True, unlabeled_test_used_for_training=False,
                  original_validation_rng_structure_preserved=True,
                  note='Only caching exact pixels/tensors. No claim of an accuracy gain or guaranteed reproduction of a public score.')
    print(f'Device={device}; GPU={torch.cuda.get_device_name(0)}; smoke={args.smoke}; Small448 native batch=12', flush=True)
    if args.smoke:
        # Default lightweight smoke uses batch4; an optional native-batch probe
        # checks the memory assumption without doing complete local training.
        count = 5 if args.smoke_batch == 4 else 20
        validation_offset = 10 if args.smoke_batch == 4 else 40
        chosen = [training[100 * label + i] for label in range(5) for i in range(count)]
        validation = [training[100 * label + validation_offset + i] for label in range(5) for i in range(4)]
        smoke_cfg = dict(cfg, epochs=1, batch_size=args.smoke_batch)
        sprint.set_seed(42)
        model, best, best_f1 = sprint.run_fold(0, chosen, validation, smoke_cfg, CLASSES, device, str(folder))
        model.load_state_dict(best)
        inference = DataLoader(sprint.PlantDS(testing[:8], 448, False), batch_size=4,
                               num_workers=0, generator=torch.Generator().manual_seed(918))
        prob, _ = sprint.predict(model, inference, device, False)
        probability_check(prob, 8)
        report.update(smoke_passed=True, smoke_batch_size=args.smoke_batch,
                      native_batch12_locally_verified=args.smoke_batch == 12,
                      smoke_labeled_train=len(chosen), smoke_validation=20, smoke_test=8,
                      smoke_diagnostic_f1=best_f1, score_is_only_a_smoke_diagnostic=True,
                      final_submission_written=False)
    else:
        sprint.main(quick=False)
        cv = json.loads((folder / 'cv_report.json').read_text(encoding='utf-8'))
        assert cv['classes'] == CLASSES and len(cv['fold_f1']) == 5
        actual = cv['cfg']
        for key in ['arch', 'img_size', 'epochs', 'batch_size', 'mix_prob', 'seed', 'tta']:
            assert actual[key] == cfg[key], f'Unexpected config override: {key}'
        assert actual['epochs'] == 40 and actual['batch_size'] == 12
        oof = np.load(folder / 'oof_prob.npy', allow_pickle=False)
        probability_check(oof, 500)
        y = np.array([item[2] for item in training])
        assert abs(float((oof.argmax(1) == y).mean()) - cv['oof_accuracy']) < 1e-12
        probabilities = []
        for fold in range(5):
            assert (folder / f'best_fold{fold}.pth').is_file()
            prob = np.load(folder / f'test_prob_fold{fold}.npy', allow_pickle=False)
            probability_check(prob, 378)
            probabilities.append(prob)
        mean = np.mean(probabilities, axis=0)
        write_csv(submission, mean, testing, template)
        report.update(complete_cv=True, teacher_oof_micro_f1=cv['oof_accuracy'],
                      teacher_oof_macro_f1=cv['oof_macro_f1'], final_submission_written=True,
                      submission_path=str(submission))
    report.update(minutes=(time.monotonic() - started) / 60,
                  peak_cuda_gb=torch.cuda.max_memory_allocated() / 1024 ** 3,
                  torch_version=torch.__version__, timm_version=timm.__version__)
    (folder / 'run_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
