"""Original supervised recipe followed by one conservative self-training round.

Only official train labels and official unlabeled test images are used. The
student has no reported OOF score: its ensemble teacher used all train labels.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sprint_refined as base

CLASSES = ['Black-grass', 'Common wheat', 'Loose Silky-bent', 'Scentless Mayweed', 'Sugar beet']
ARCH = 'convnext_small.fb_in22k_ft_in1k'


class BudgetExceeded(RuntimeError):
    pass


def check_budget(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExceeded('Training budget exhausted')


def check_prob(prob, count):
    if prob.shape != (count, 5) or not np.isfinite(prob).all() or (prob < 0).any():
        raise ValueError('Invalid probabilities')
    if not np.allclose(prob.sum(1), 1, atol=.002):
        raise ValueError('Class probabilities must sum to one')


def discover(root):
    return base.read_course_data(root, CLASSES)


def select_pseudo(fold_prob, threshold=.9, cap=50):
    if fold_prob.ndim != 3 or fold_prob.shape[0] != 5 or fold_prob.shape[2] != 5:
        raise ValueError('Selection requires five-fold predictions')
    for probability in fold_prob:
        check_prob(probability, fold_prob.shape[1])
    mean = fold_prob.mean(0)
    prediction = mean.argmax(1)
    confidence = mean.max(1)
    unanimous = (fold_prob.argmax(2) == prediction[None]).all(0)
    eligible = (confidence >= threshold) & unanimous
    chosen = []
    for label in range(5):
        candidates = np.flatnonzero(eligible & (prediction == label))
        order = np.argsort(-confidence[candidates], kind='stable')
        chosen.extend(candidates[order][:cap].tolist())
    return np.array(sorted(chosen), dtype=np.int64), mean, eligible


class SelfTrainingDataset(Dataset):
    def __init__(self, labeled_items, test_items, selected, soft_labels, size, pseudo_weight):
        items = list(labeled_items)
        targets = []
        for _, _, label in labeled_items:
            target = np.full(6, .1 / 5, dtype=np.float32)
            target[label] += .9
            target[5] = 1.
            targets.append(target)
        for index in selected:
            # No test label is read: use the detached teacher distribution only.
            path = test_items[int(index)][0]
            items.append((path, path, 0))
            targets.append(np.r_[soft_labels[index], np.float32(pseudo_weight)].astype(np.float32))
        self.images = base.PlantDS(items, size, train=True, aug_level='strong')
        if self.images.ra is None:
            raise RuntimeError('Original RandAugment must be available')
        self.targets = torch.tensor(np.stack(targets), dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, _, _ = self.images[index]
        return image, self.targets[index], index


def weighted_soft_loss(logits, target):
    probabilities = target[:, :5].detach()
    weights = target[:, 5].detach()
    return (-(probabilities * F.log_softmax(logits.float(), dim=1)).sum(1) * weights).mean()


def finetune(model, dataset, cfg, epochs, seed, deadline, workers):
    # This is a new stage. Its RNG is independent of the original teacher stage.
    base.set_seed(seed)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=workers, persistent_workers=workers > 0,
                        generator=torch.Generator().manual_seed(seed),
                        drop_last=True, pin_memory=True)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(base.param_groups(model, 3e-5, cfg['weight_decay']))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    ema = base.EMA(model, cfg['ema_decay'])
    total_steps = max(1, len(loader) * epochs)
    successful = 0
    histories = []
    for epoch in range(epochs):
        model.train()
        loss_sum, count, mixed, overflow = 0., 0, 0, 0
        for images, targets, _ in loader:
            check_budget(deadline)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                loss, _, _, lam = base.mixup_cutmix_loss(weighted_soft_loss, model, images, targets, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite student loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            finite = bool(torch.isfinite(norm))
            if not finite and not scaler.is_enabled():
                raise FloatingPointError('Non-finite student gradient')
            scaler.step(optimizer)
            scaler.update()
            if finite:
                successful += 1
                ema.update(model, successful)
                multiplier = .1 + .9 * .5 * (1 + math.cos(math.pi * min(1., successful / total_steps)))
                optimizer.param_groups[0]['lr'] = 9e-6 * multiplier
                optimizer.param_groups[1]['lr'] = 3e-5 * multiplier
            else:
                overflow += 1
            loss_sum += float(loss.detach()) * len(images)
            count += len(images)
            mixed += int(lam != 1.)
        if not successful:
            raise FloatingPointError('No successful student updates')
        record = dict(epoch=epoch + 1, loss=loss_sum / count, mixed_batches=mixed,
                      amp_overflow_batches=overflow, successful_updates=successful)
        histories.append(record)
        print(f'Student {seed} ep {epoch + 1}/{epochs}: loss={record["loss"]:.4f}', flush=True)
    del loader
    return ema, histories


def predict_average(model, ema, loader, device):
    raw, _ = base.predict(model, loader, device, tta=False)
    ema.apply_to(model)
    try:
        averaged, _ = base.predict(model, loader, device, tta=False)
    finally:
        ema.restore(model)
    return .5 * raw + .5 * averaged


def write_submission(path, probability, test_items, template_ids):
    check_prob(probability, 378)
    predictions = {Path(item[0]).name: CLASSES[int(probability[i].argmax())]
                   for i, item in enumerate(test_items)}
    if len(predictions) != 378 or set(predictions) != set(template_ids):
        raise ValueError('Submission IDs mismatch')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ID', 'Category'])
        writer.writerows((image_id, predictions[image_id]) for image_id in template_ids)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--submission-path', help='Defaults to submission.csv inside the run directory')
    parser.add_argument('--threshold', type=float, default=.9)
    parser.add_argument('--pseudo-weight', type=float, default=.2)
    parser.add_argument('--cap-per-class', type=int, default=50)
    parser.add_argument('--student-epochs', type=int, default=5)
    parser.add_argument('--blend', type=float, default=.5)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-minutes', type=float, default=110.)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--allow-unlabeled-test-training', action='store_true',
                        help='Enable the exploratory run after confirming the course permits this use')
    args = parser.parse_args()
    if not args.allow_unlabeled_test_training:
        parser.error('The exploratory run requires --allow-unlabeled-test-training')
    if not 0 < args.pseudo_weight <= .5 or not .8 <= args.threshold < 1 or not 0 <= args.blend <= 1:
        parser.error('Invalid confidence, loss weight, or blend')
    if not 1 <= args.student_epochs <= 8 or not 1 <= args.max_minutes <= 112 or args.cap_per_class < 1:
        parser.error('Invalid epoch count or time budget')
    started = time.monotonic()
    deadline = started + args.max_minutes * 60
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('Use a CUDA GPU for this full-resolution workflow')
    import timm
    # Preserve the filename suffix: HF cache snapshots are symlinks on Linux.
    # Resolving one produces a suffixless blob path, which timm can mistake for
    # a PyTorch pickle instead of a safetensors file.
    weights = Path(args.weights).absolute()
    if not weights.is_file():
        raise FileNotFoundError('Approved ImageNet pretrained weights are required')

    def build_teacher(cfg, count):
        model = timm.create_model(ARCH, pretrained=True, num_classes=count,
                                  drop_rate=cfg['drop_rate'], drop_path_rate=cfg['drop_path_rate'],
                                  pretrained_cfg_overlay={'file': str(weights)})
        model.register_forward_pre_hook(lambda _model, _inputs: check_budget(deadline))
        return model

    # Only pin weight loading and enforce the deadline. Original training,
    # validation workers, scheduler, EMA and checkpoint selection are untouched.
    base.build_model = build_teacher
    folder = Path(args.out_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    teacher_dir = folder / 'teacher'
    teacher_dir.mkdir(exist_ok=True)
    if not args.smoke and any(teacher_dir.iterdir()):
        raise RuntimeError('Use a fresh output folder to avoid stale teacher outputs')
    submission = Path(args.submission_path).absolute() if args.submission_path else folder / 'submission.csv'
    if not args.smoke and submission.exists():
        raise FileExistsError(f'Choose a new submission path: {submission}')
    train_dir, test_dir, train_items, test_items, template_ids = discover(args.data_root)
    cfg = dict(base.CFG)
    cfg.update(out_dir=str(teacher_dir), num_workers=args.workers)
    # Keep the .9708 teacher's entire recipe, including persistent val workers.
    base.CFG.update(cfg)
    base.find_dirs = lambda *a, **k: (str(train_dir), str(test_dir))
    np.save(folder / 'test_ids.npy', np.array([Path(item[0]).name for item in test_items]))
    input_hash = hashlib.sha256()
    for item in train_items + test_items:
        input_hash.update(Path(item[0]).name.encode())
        input_hash.update(hashlib.sha256(Path(item[0]).read_bytes()).digest())
    print(f'Device={device}; GPU={torch.cuda.get_device_name(0)}; smoke={args.smoke}; official train=500 test=378', flush=True)
    report = dict(smoke=args.smoke, original_teacher_recipe_preserved=not args.smoke,
                  official_labeled_count=500, official_unlabeled_count=378,
                  input_sha256=input_hash.hexdigest(), exploratory_unlabeled_test_training=True,
                  teacher_recipe='reference Small448/b12/40epochs/5fold/seed42/mix_prob0.3',
                  threshold=args.threshold, pseudo_weight=args.pseudo_weight,
                  cap_per_class=args.cap_per_class, student_epochs=args.student_epochs,
                  student_uses_all_500_train_labels=not args.smoke,
                  stage2_validation_score=None,
                  stage2_validation_note='Teacher ensemble used every training label. Student training uses all 500 labeled images. No student OOF score is valid or reported.',
                  hidden_test_labels_used=False, score_improvement_verified=False)
    if args.smoke:
        from sklearn.model_selection import StratifiedKFold
        base.set_seed(42)
        y = np.array([item[2] for item in train_items])
        train_index, val_index = next(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(500), y))
        # Native batch12 smoke: 100 true-label training, 20 disjoint validation.
        labeled = [train_items[100 * label + i] for label in range(5) for i in range(20)]
        validation = [train_items[100 * label + 40 + i] for label in range(5) for i in range(4)]
        smoke_cfg = dict(cfg, epochs=1)
        model, best, _ = base.run_fold(0, labeled, validation, smoke_cfg, CLASSES, device, str(teacher_dir))
        model.load_state_dict(best)
        inference = DataLoader(base.PlantDS(test_items[:8], cfg['img_size'], False),
                               batch_size=12, num_workers=0, generator=torch.Generator().manual_seed(555))
        teacher_probability, _ = base.predict(model, inference, device, tta=False)
        # Deliberately force eight labels to exercise the code path. Smoke labels
        # are never eligible for a competition submission or a quality claim.
        selected = np.arange(8, dtype=np.int64)
        report.update(smoke_forced_pseudo_count=8, pseudo_selection_forced_for_smoke=True,
                      smoke_labeled_count=100, smoke_validation_count=20, smoke_batch_size=12)
        dataset = SelfTrainingDataset(labeled, test_items[:8], selected, teacher_probability,
                                      cfg['img_size'], args.pseudo_weight)
        ema, history = finetune(model, dataset, cfg, 1, 1042, deadline, args.workers)
        probability = predict_average(model, ema, inference, device)
        check_prob(probability, 8)
        report.update(student_history=history, smoke_passed=True, smoke_test_predictions=8,
                      final_submission_written=False)
    else:
        base.main(quick=False)
        baseline_report = json.loads((teacher_dir / 'cv_report.json').read_text(encoding='utf-8'))
        report['teacher_oof_micro_f1'] = baseline_report['oof_accuracy']
        report['teacher_oof_macro_f1'] = baseline_report['oof_macro_f1']
        assert baseline_report['classes'] == CLASSES and len(baseline_report['fold_f1']) == 5
        for key in ['arch', 'img_size', 'epochs', 'batch_size', 'mix_prob', 'seed', 'tta']:
            assert baseline_report['cfg'][key] == cfg[key]
        check_prob(np.load(teacher_dir / 'oof_prob.npy', allow_pickle=False), 500)
        fold_probability = np.stack([np.load(teacher_dir / f'test_prob_fold{i}.npy', allow_pickle=False) for i in range(5)])
        selected, teacher_probability, eligible = select_pseudo(fold_probability, args.threshold, args.cap_per_class)
        np.savez_compressed(folder / 'pseudo_selection.npz', selected=selected,
                            probabilities=teacher_probability, eligible=eligible,
                            test_ids=np.array([Path(item[0]).name for item in test_items]))
        report.update(pseudo_eligible_before_cap=int(eligible.sum()), pseudo_selected=int(len(selected)),
                      pseudo_counts={name: int((teacher_probability[selected].argmax(1) == i).sum())
                                     for i, name in enumerate(CLASSES)})
        # A complete teacher submission is available even if the added stage hits
        # its time budget. The fallback is explicitly reported, never disguised.
        write_submission(submission, teacher_probability, test_items, template_ids)
        report['fallback_to_teacher'] = not bool(len(selected))
        student_probability = []
        if len(selected):
            dataset = SelfTrainingDataset(train_items, test_items, selected, teacher_probability,
                                          cfg['img_size'], args.pseudo_weight)
            inference = DataLoader(base.PlantDS(test_items, cfg['img_size'], False),
                                   batch_size=12, num_workers=0, pin_memory=True,
                                   generator=torch.Generator().manual_seed(555))
            try:
                for fold in range(5):
                    check_budget(deadline)
                    model = timm.create_model(ARCH, pretrained=False, num_classes=5,
                                              drop_rate=.1, drop_path_rate=.1).to(device)
                    checkpoint = torch.load(teacher_dir / f'best_fold{fold}.pth', map_location='cpu', weights_only=True)
                    assert checkpoint['classes'] == CLASSES and checkpoint['arch'] == ARCH and checkpoint['img_size'] == 448
                    model.load_state_dict(checkpoint['state'])
                    del checkpoint
                    ema, history = finetune(model, dataset, cfg, args.student_epochs, 1042 + fold, deadline, args.workers)
                    check_budget(deadline)
                    probability = predict_average(model, ema, inference, device)
                    check_prob(probability, 378)
                    student_probability.append(probability)
                    np.save(folder / f'student_test_fold{fold}.npy', probability)
                    (folder / f'student_fold{fold}_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
                    del model, ema
                    gc.collect()
                    torch.cuda.empty_cache()
            except BudgetExceeded:
                report['fallback_to_teacher'] = True
                print('Time budget reached; retaining complete teacher submission.', flush=True)
        report['completed_student_models'] = len(student_probability)
        report['pseudo_stage_complete'] = len(student_probability) == 5
        if report['pseudo_stage_complete']:
            student_mean = np.mean(student_probability, axis=0)
            final = (1 - args.blend) * teacher_probability + args.blend * student_mean
            write_submission(submission, final, test_items, template_ids)
            report.update(fallback_to_teacher=False, final_student_blend=args.blend)
        report.update(final_submission_written=True, submission_path=str(submission))
    report.update(minutes=(time.monotonic() - started) / 60,
                  torch_version=torch.__version__, timm_version=timm.__version__,
                  peak_cuda_gb=torch.cuda.max_memory_allocated() / 1024 ** 3)
    (folder / 'run_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
