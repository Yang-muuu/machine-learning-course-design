"""表情分类课设：重建数据、逐折训练、两种融合、提交文件核验。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ORDER = ('convnext_eca', 'dinov2_small', 'ddam_drop', 'dan', 'b2_native', 'ddam_old', 'ddam_sam')
WEIGHTS = (.2, .2, .4, .2)
GRID = (.0003, .001, .003, .01)
ROWS, TEST_ROWS = 28165, 7065
TRAIN_SHA = '1741ed6a65e0abe4c5ffa5e163a94bc38db432d0e88e41a2306b72d539358c57'
SPLIT_SHA = '1230217c9b9e74371628f249bebc63bb1d0443e16c0b18e46d596ac470a080c7'


def read(path):
    return Path(path).read_text(encoding='utf-8')


def load(path):
    return json.loads(read(path))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def ensure(ok, message):
    if not ok:
        raise ValueError(message)


def check_repo():
    files = load(ROOT/'MANIFEST.json')['files']
    for name, expected in files.items():
        p = ROOT/name
        ensure(p.is_file() and sha(p) == expected, f'文件缺失或不符：{name}')
    print(f'代码和实验记录核验通过：{len(files)} 个文件', flush=True)


def checked_weight(model):
    asset = load(ROOT/'configs/pretrained.json')[model]
    p = ROOT/'local/weights'/asset['filename']
    ensure(p.is_file(), f'缺少 {p}；请先按 README 准备初始化权重')
    ensure(p.stat().st_size == asset['bytes'] and sha(p) == asset['sha256'], f'初始化权重指纹不符：{p}')
    return p, asset


def read_csv(path):
    return pd.read_csv(path, dtype={'ID': str}, keep_default_na=False)


def same_rows(left, right, fields):
    """对齐逐列值；容许 CSV 重读时 pandas / NumPy 整数位宽不同。"""
    return len(left) == len(right) and all(np.array_equal(left[k].to_numpy(), right[k].to_numpy()) for k in fields)


def submission(path, *, template=False):
    result = read_csv(path)
    ensure(result.columns.tolist() == ['ID', 'Emotion'], f'CSV 列必须为 ID,Emotion：{path}')
    ensure(len(result) == TEST_ROWS and result.ID.is_unique and result.ID.ne('').all(), f'测试行数或ID不符：{path}')
    if not template:
        ensure(result.Emotion.isin(range(6)).all(), f'Emotion 必须为 0 至 5 的整数：{path}')
    return result


def check_data(work, *, pixels=False):
    cache = work/'cache'
    train = read_csv(cache/'train_manifest.csv')
    test = read_csv(cache/'test_manifest.csv')
    sample = submission(cache/'sample_submission.csv', template=True)
    ensure(len(train) == ROWS and train.ID.is_unique and test.ID.equals(sample.ID), '清洗后的训练/测试清单与实验不符')
    for frame, filename in [(train,'train_images.npy'), (test,'test_images.npy')]:
        images = np.load(cache/filename, mmap_mode='r', allow_pickle=False)
        ensure(images.dtype == np.uint8 and images.shape == (len(frame),48,48), f'灰度缓存形状错误：{filename}')
        if pixels:
            ensure([hashlib.sha256(img.tobytes()).hexdigest() for img in images] == frame.pixel_hash.tolist(), '缓存像素指纹不符')
    signature = train[['ID','label','pixel_hash']].sort_values('ID').to_csv(index=False,lineterminator='\n').encode()
    ensure(hashlib.sha256(signature).hexdigest() == TRAIN_SHA, '训练集像素/标签签名不符，请使用课程原数据')
    ensure(np.array_equal(train.array_index, np.arange(ROWS)), '训练缓存索引不连续')
    code = read(ROOT/'configs/folds.txt').strip()
    ensure(len(code) == ROWS and set(code) == set('01234'), '折号文件无效')
    train['validation_fold'] = np.fromiter((int(x) for x in code), dtype=int, count=ROWS)
    canonical = train[['ID','label','validation_fold']].to_csv(index=False,lineterminator='\n').encode()
    ensure(hashlib.sha256(canonical).hexdigest() == SPLIT_SHA, '固定折与缓存顺序不匹配')
    ensure(train.groupby('pixel_hash').validation_fold.nunique().max() == 1, '相同像素跨折')
    return train, sample


def prepare(source, work):
    ensure(source.exists(), f'请先准备原始比赛数据：{source}')
    sys.path.insert(0, str(ROOT/'engine/vendor/convnext_eca'))
    from prepare_data import prepare_data
    prepare_data(source, work/'cache')
    fixed, _ = check_data(work, pixels=True)
    fixed.to_csv(work/'reference_split.csv', index=False, lineterminator='\n')
    print('数据签名、像素和固定分折核验通过；验证折样本数：', fixed.validation_fold.value_counts().sort_index().tolist(), flush=True)


def prob(p, size):
    ensure(p.shape == (size,6) and np.isfinite(p).all() and (p >= 0).all()
           and np.allclose(p.sum(1), 1, atol=1e-5), '六类概率形状或数值无效')
    return p


def fold_result(work, name, fold, fixed):
    run = work/'models'/name/'runs'/f'fold_{fold}'
    record = load(run/'run_summary.json')
    ensure(not record['smoke'] and record['fold'] == fold, f'不是正式验证折：{run}')
    ensure(record['checkpoint_sha256'] == sha(run/'best.pt'), f'权重指纹不符：{run}')
    ensure(record['source_hashes'] == load(ROOT/'configs/source_hashes.json')[name], f'源码版本不符：{run}')
    ensure(record['reload_max_difference'] <= 1e-5, f'训练后重载差异过大：{run}')
    expected = fixed.loc[fixed.validation_fold.eq(fold)]
    rows = read_csv(run/'validation_rows.csv')
    fields = ['ID','label','pixel_hash','array_index']
    ensure(same_rows(rows, expected, fields), f'验证预测ID顺序错配：{run}')
    cfg = load(run/'config.json')
    for key, v in load(ROOT/'configs'/f'{name}.json').items():
        ensure(cfg[key] == v, f'历史配置不符：{name}/{key}')
    val = prob(np.load(run/'validation_probabilities.npy',allow_pickle=False), len(rows))
    test = prob(np.load(run/'test_probabilities.npy',allow_pickle=False), TEST_ROWS)
    ensure(np.array_equal(rows.prediction, val.argmax(1)), f'验证标签和概率不符：{run}')
    ensure(abs((val.argmax(1) == rows.label.to_numpy()).mean() - record['accuracy']) < 1e-10,
           f'验证准确率与摘要不符：{run}')
    return val, test


def train(args, *, smoke):
    check_repo()
    fixed, _ = check_data(args.work)
    expected = read_csv(args.work/'reference_split.csv')
    # 灰度标准差只作诊断，CSV 浮点往返不影响样本、标签或折号。
    ensure(same_rows(fixed, expected, ['ID','label','pixel_hash','array_index','validation_fold']),
           '运行目录的训练清单有变，请重新执行 prepare')
    default_models = ORDER[:4] if args.scheme == 'weighted' else ORDER
    names = list(dict.fromkeys(args.model or default_models))
    folds = args.folds[:1] if smoke else args.folds
    ensure(len(set(folds)) == len(folds), '验证折不能重复')
    for name in names:
        checked_weight(name)
        for fold in folds:
            if not smoke:
                run = args.work/'models'/name/'runs'/f'fold_{fold}'
                if (run/'run_summary.json').is_file():
                    fold_result(args.work,name,fold,fixed)
                    print(f'已验证完成，跳过 {name} fold={fold}', flush=True)
                    continue
                ensure(not run.exists() or not any(run.iterdir()), f'发现未完成折：{run}；请先备份该目录')
            env = os.environ.copy()
            env.update(PYTHONUNBUFFERED='1', PYTHONUTF8='1')
            if args.device == 'cpu':
                env['CUDA_VISIBLE_DEVICES'] = ''
            subprocess.run([sys.executable,'-u',str(ROOT/'engine/worker.py'),'--model',name,
                '--work',str(args.work),'--fold',str(fold),'--device',args.device,'--mode','smoke' if smoke else 'train'],
                check=True, env=env)
            if not smoke:
                fold_result(args.work,name,fold,fixed)


def pack_model(work, name, fixed):
    val = np.empty((ROWS,6),dtype=np.float64)
    bank = []
    for fold in range(5):
        pred, test = fold_result(work,name,fold,fixed)
        val[fixed.validation_fold.to_numpy() == fold] = pred
        bank.append(test)
    # 各折原概率为 float32。保留历史的五折平均精度，再转 float64 供跨模型融合。
    return val, np.stack(bank).mean(0).astype(np.float32).astype(np.float64)


def weighted(bank):
    return sum(w * p for w,p in zip(WEIGHTS,bank[:4]))


def features(bank):
    return np.concatenate([np.log(np.maximum(p, 1e-6)) for p in bank], axis=1)


def apply_stack(bank, params):
    x = features(bank)
    ensure(np.array_equal(params['classes'],np.arange(6)) and params['coef'].shape == (6,42), '融合参数类别顺序或形状错误')
    logits = ((x-params['mean'])/params['scale']) @ params['coef'].T + params['intercept']
    logits -= logits.max(axis=1,keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1,keepdims=True)


def metric(y, p):
    prob(p,len(y))
    cm = np.zeros((6,6), dtype=np.int64)
    np.add.at(cm,(y,p.argmax(1)),1)
    return {'correct':int(cm.trace()),'rows':len(y),'accuracy':float(cm.trace()/len(y)),
            'macro_f1':float((2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1)).mean()),
            'log_loss':float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-15)).mean()),'confusion':cm.tolist()}


def fit_stack(oo, tt, fixed, out):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    x, y, folds = features(oo), fixed.label.to_numpy(int), fixed.validation_fold.to_numpy(int)
    baseline = metric(y,weighted(oo))
    reports = []
    with threadpool_limits(limits=1):
        for c in GRID:
            p = np.empty((len(y),6),dtype=np.float64)
            for fold in range(5):
                mask = folds == fold
                scaler = StandardScaler().fit(x[~mask])
                fit = LogisticRegression(C=c,solver='lbfgs',max_iter=1000,tol=1e-7)
                fit.fit(scaler.transform(x[~mask]), y[~mask])
                ensure(fit.n_iter_.max() < 1000 and np.array_equal(fit.classes_,np.arange(6)), '元分类器未收敛或类别缺失')
                p[mask] = fit.predict_proba(scaler.transform(x[mask]))
            reports.append(({'C':c, **metric(y,p)}, p))
        eligible = [r for r in reports if r[0]['macro_f1'] >= baseline['macro_f1']-.003]
        ensure(bool(eligible), '此次重训没有达到 Macro F1 门槛的堆叠候选')
        (chosen, cv) = max(eligible,key=lambda r:(r[0]['accuracy'],-r[0]['log_loss']))
        scaler = StandardScaler().fit(x)
        fit = LogisticRegression(C=chosen['C'],solver='lbfgs',max_iter=1000,tol=1e-7)
        fit.fit(scaler.transform(x), y)
        ensure(fit.n_iter_.max() < 1000, '最终元分类器未收敛')
    params = dict(mean=scaler.mean_,scale=scaler.scale_,coef=fit.coef_,intercept=fit.intercept_,classes=fit.classes_)
    np.savez(out/'stacking_parameters.npz',**params)
    dump(out/'stacking_grid.json',{'candidates':[r[0] for r in reports], 'selected_C':chosen['C'],
                                  'scope':'meta crossfit on fixed base OOF, not nested training of base models'})
    return cv, apply_stack(tt,params)


def export(out, sample, pred):
    result = sample.copy()
    result['Emotion'] = prob(pred,TEST_ROWS).argmax(1)
    result.to_csv(out/'submission.csv',index=False,lineterminator='\r\n')
    np.save(out/'test_probabilities.npy',pred,allow_pickle=False)
    print('生成:',out/'submission.csv','SHA256:',sha(out/'submission.csv'),flush=True)


def fuse(args):
    fixed, sample = check_data(args.work)
    names = ORDER[:4] if args.scheme=='weighted' else ORDER
    oo,tt = map(list,zip(*(pack_model(args.work,n,fixed) for n in names)))
    args.output.mkdir(parents=True,exist_ok=False)
    if args.scheme=='weighted':
        cv,p = weighted(oo),weighted(tt)
    else:
        cv,p = fit_stack(oo,tt,fixed,args.output)
    dump(args.output/'oof_metrics.json',metric(fixed.label.to_numpy(int),cv))
    export(args.output,sample,p)


def replay(args):
    sample = submission(args.sample,template=True)
    names = ORDER[:4] if args.scheme=='weighted' else ORDER
    bank = [prob(np.load(args.predictions/f'{n}_test.npy',allow_pickle=False),TEST_ROWS) for n in names]
    if args.scheme=='weighted':
        p = weighted(bank)
    else:
        ensure(args.parameters is not None, '堆叠重放需要 --parameters')
        with np.load(args.parameters,allow_pickle=False) as z:
            p = apply_stack(bank,z)
    args.output.mkdir(parents=True,exist_ok=False)
    export(args.output,sample,p)
    expected = load(ROOT/'evidence/summary.json')['submissions']['6' if args.scheme=='weighted' else '5']['sha256']
    ensure(sha(args.output/'submission.csv') == expected, '重放结果不等于原提交，请检查历史概率和提交模板来源')


def compare(args):
    a,b = submission(args.first),submission(args.second)
    ensure(set(a.ID) == set(b.ID),'两份 CSV 的 ID 集合不同')
    a = a.set_index('ID').Emotion
    b = b.set_index('ID').Emotion.reindex(a.index)
    diff = int((a != b).sum())
    groups = Counter(tuple(sorted((int(x),int(y)))) for x,y in zip(a,b) if x != y)
    print(json.dumps({'rows':len(a),'different_predictions':diff,'rate':diff/len(a),
        'pair_counts':[[*k,v] for k,v in groups.most_common()], 'first_sha256':sha(args.first),
        'second_sha256':sha(args.second)},ensure_ascii=False,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command',required=True)
    subs.add_parser('verify')
    p = subs.add_parser('prepare'); p.add_argument('--source',type=Path,default=ROOT/'local/2026task2.zip')
    p.add_argument('--work',type=Path,default=ROOT/'work')
    for command in ['train','smoke']:
        p=subs.add_parser(command); p.add_argument('--scheme',choices=['weighted','stacked'],default='weighted')
        p.add_argument('--model',nargs='+',choices=ORDER); p.add_argument('--folds',nargs='+',type=int,choices=range(5),default=list(range(5)))
        p.add_argument('--device',default='cpu' if command=='smoke' else 'cuda:0'); p.add_argument('--work',type=Path,default=ROOT/'work')
    for command in ['fuse','replay']:
        p=subs.add_parser(command); p.add_argument('--scheme',choices=['weighted','stacked'],required=True)
        p.add_argument('--output',type=Path,required=True)
        if command=='fuse':p.add_argument('--work',type=Path,default=ROOT/'work')
        else:
            p.add_argument('--predictions',type=Path,required=True);p.add_argument('--sample',type=Path,required=True)
            p.add_argument('--parameters',type=Path)
    p=subs.add_parser('compare-submissions');p.add_argument('--first',type=Path,required=True);p.add_argument('--second',type=Path,required=True)
    args=parser.parse_args()
    if hasattr(args,'work'):args.work=args.work.resolve()
    if args.command=='verify':check_repo()
    elif args.command=='prepare':prepare(args.source,args.work)
    elif args.command in ['train','smoke']:train(args,smoke=args.command=='smoke')
    elif args.command=='fuse':fuse(args)
    elif args.command=='replay':replay(args)
    else:compare(args)


if __name__=='__main__':main()
