"""Frozen HRNet/RFM with paired luminance/green contrast, original fivefold."""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import timm

from contrast_models import (VARIANTS,AdaptModel,ImageBank,file_hash,frozen_hash,
    load_saved,make_batch,objective,predict,score,seed_everything,state_hash,trainable_parameters,REG_CONFIG)


def initialize(seed):
    seed_everything(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def worker(args,smoke=False):
    ImageBank.variant=REG_CONFIG[args.variant]['contrast']
    assert torch.cuda.is_available()
    seed=args.seed+args.fold
    initialize(seed)
    device=torch.device(args.device)
    torch.cuda.set_device(device)
    train=pd.read_csv(args.artifacts/'train.csv')
    test=pd.read_csv(args.artifacts/'test.csv')
    tr=np.flatnonzero(train.fold.values!=args.fold)
    va=np.flatnonzero(train.fold.values==args.fold)
    assert set(train.iloc[tr].group).isdisjoint(train.iloc[va].group)
    train_bank=ImageBank(train,args.data,'train',device)
    test_bank=ImageBank(test,args.data,'test',device)
    model=AdaptModel(args.variant,args.assets,seed).to(device)
    fixed_hash=frozen_hash(model)
    initial_head=state_hash(model.head)
    initial_encoder=state_hash(model.encoder)
    trainable=trainable_parameters(model)
    head=[p for n,p in trainable.items() if n.startswith('head.')]
    encoder=[p for n,p in trainable.items() if n.startswith('encoder.')]
    groups=[dict(params=head,lr=args.lr)]
    if encoder: groups.append(dict(params=encoder,lr=args.encoder_lr))
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=bool(encoder),init_scale=1024,growth_interval=2000)
    seed_everything(seed)
    folder=args.output/args.variant/f'fold{args.fold}'
    if smoke:
        model.train()
        model.set_epoch(6)
        ids=tr[:args.batch_size]
        images,xy,boxes,theta=make_batch(train_bank,ids,augment=True)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        loss=objective(model(images,amp=True),xy,boxes,theta,train_bank,ids,)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert sum(float(p.grad.abs().sum()) for p in head)>0
        if encoder:
            assert sum(float(p.grad.abs().sum()) for p in encoder)>0
        norm=torch.nn.utils.clip_grad_norm_(trainable.values(),5)
        assert torch.isfinite(norm)
        scaler.step(optimizer)
        scaler.update()
        assert frozen_hash(model)==fixed_hash
        assert state_hash(model.head)!=initial_head
        assert (state_hash(model.encoder)!=initial_encoder)==bool(encoder)
        result=dict(status='PASS',variant=args.variant,loss=float(loss.detach()),finite_gradients=True,
            actual_head_update=True,actual_stage4_update=bool(encoder),all_bn_and_frozen_state_unchanged=True,
            initial_frozen_state_sha256=fixed_hash,strict_official_load=model.encoder.load_info,
            initial_encoder_sha256=initial_encoder,head_parameters=sum(p.numel() for p in head),
            encoder_trainable_parameters=sum(p.numel() for p in encoder),batch_size=len(ids),
            peak_cuda_allocated_mb=torch.cuda.max_memory_allocated(device)/1024**2,
            used_training_ids=train.iloc[ids].image_id.tolist(),validation_ids_excluded=train.iloc[va].image_id.tolist(),
            discarded_before_formal_training=True,gpu=torch.cuda.get_device_name(device))
        pd.concat([train_bank.geometry,test_bank.geometry]).to_csv(args.output/'image_geometry.csv',index=False)
        del model,optimizer,scaler,train_bank,test_bank
        gc.collect();torch.cuda.empty_cache()
        return result
    folder.mkdir(parents=True,exist_ok=True)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    ema_parameters=dict(ema.named_parameters())
    history=[]
    started=time.perf_counter()
    print(f'START {args.variant} fold={args.fold} head={sum(p.numel() for p in head)} encoder_trainable={sum(p.numel() for p in encoder)}',flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();model.set_epoch(epoch)
        factor=min(1,epoch/5)*(.05+.95*.5*(1+math.cos(math.pi*max(0,epoch-5)/(args.schedule_epochs-5))))
        optimizer.param_groups[0]['lr']=args.lr*factor
        if encoder: optimizer.param_groups[1]['lr']=args.encoder_lr*factor if epoch>5 else 0
        losses,norms=[],[]
        skipped=0
        for ids in np.array_split(np.random.permutation(tr),math.ceil(len(tr)/args.batch_size)):
            images,xy,boxes,theta=make_batch(train_bank,ids,augment=True)
            optimizer.zero_grad(set_to_none=True)
            loss=objective(model(images,amp=True),xy,boxes,theta,train_bank,ids,)
            if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss {args.variant} fold={args.fold} epoch={epoch}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(trainable.values(),5)
            previous_scale=scaler.get_scale()
            if not torch.isfinite(norm) and not scaler.is_enabled(): raise RuntimeError('Nonfinite gradient')
            scaler.step(optimizer);scaler.update()
            step_skipped=scaler.get_scale()<previous_scale
            if step_skipped: skipped+=1
            else:
                with torch.no_grad():
                    for name,current in trainable.items(): ema_parameters[name].lerp_(current.detach(),.03)
            losses.append(float(loss.detach()))
            if torch.isfinite(norm): norms.append(float(norm))
        assert norms and skipped<math.ceil(len(tr)/args.batch_size)
        record=dict(variant=args.variant,fold=args.fold,epoch=epoch,loss=float(np.mean(losses)),
            gradient_norm=float(np.mean(norms)),head_lr=optimizer.param_groups[0]['lr'],
            encoder_lr=optimizer.param_groups[1]['lr'] if encoder else 0,
            skipped_steps=skipped,elapsed_seconds=time.perf_counter()-started)
        if epoch==1 or epoch%10==0:
            identity,tta=predict(ema,train_bank,va,args.batch_size)
            truth=train.iloc[va][['x','y']].values
            record.update(val_rmse=score(truth,identity)['rmse_pixels'],tta_val_rmse=score(truth,tta)['rmse_pixels'])
            print(f"{args.variant} fold={args.fold} epoch={epoch:02d}/{args.epochs} identity_RMSE={record['val_rmse']:.3f} tta={record['tta_val_rmse']:.3f} time={record['elapsed_seconds']:.1f}s",flush=True)
        history.append(record)
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        if epoch in [20,40]:
            assert frozen_hash(model)==fixed_hash
            dest=folder/f'epoch{epoch}'
            dest.mkdir(exist_ok=True)
            checkpoint=dict(model_name=REG_CONFIG[args.variant]['model'],variant=args.variant,fold=args.fold,seed=seed,
                completed_epochs=epoch,schedule_epochs=args.schedule_epochs,
                state_dict={k:v.detach().cpu().clone() for k,v in ema.state_dict().items()},
                frozen_state_sha256=fixed_hash,initial_head_sha256=initial_head,
                initial_encoder_sha256=initial_encoder,source_file_sha256=model.encoder.provenance['sha256'],
                source_load_info=model.encoder.load_info,train_ids=train.iloc[tr].image_id.tolist(),
                validation_ids=train.iloc[va].image_id.tolist())
            torch.save(checkpoint,dest/'model.pt')
            manifest={k:v for k,v in checkpoint.items() if k!='state_dict'}
            manifest.update(checkpoint_sha256=file_hash(dest/'model.pt'),
                head_parameters=sum(p.numel() for p in head),encoder_trainable_parameters=sum(p.numel() for p in encoder),
                actual_head_update=state_hash(ema.head)!=initial_head,
                actual_stage4_update=state_hash(ema.encoder)!=initial_encoder,
                all_bn_and_frozen_state_unchanged=True,primary_metric='identity',
                selection='fixed epoch20 full EMA; epoch40 diagnostic; never per-fold best')
            assert manifest['actual_head_update'] and manifest['actual_stage4_update']==bool(encoder)
            (dest/'training_manifest.json').write_text(json.dumps(manifest,indent=2))
            del checkpoint
    del model,ema,optimizer,scaler,trainable,ema_parameters,head,encoder
    gc.collect();torch.cuda.empty_cache()
    for epoch in [20,40]:
        dest=folder/f'epoch{epoch}'
        model,saved=load_saved(dest/'model.pt',args.assets,device)
        identity,tta=predict(model,train_bank,va,args.batch_size)
        oof=train.iloc[va][['image_id','filename','fold','group','width','height','x','y']].copy()
        oof['identity_x'],oof['identity_y']=identity[:,0],identity[:,1]
        oof['pred_x'],oof['pred_y']=tta[:,0],tta[:,1]
        oof.to_csv(dest/'oof.csv',index=False)
        identity,tta=predict(model,test_bank,np.arange(len(test)),args.batch_size)
        table=test.copy();table['fold']=args.fold
        table['identity_x'],table['identity_y']=identity[:,0],identity[:,1]
        table['pred_x'],table['pred_y']=tta[:,0],tta[:,1]
        table.to_csv(dest/'test.csv',index=False)
        del model,saved
        gc.collect();torch.cuda.empty_cache()
    print(f'COMPLETE {args.variant} fold={args.fold}: epoch20/40 full EMA checkpoints',flush=True)


def run(args):
    assert torch.cuda.is_available()
    variants=[v for v in VARIANTS if args.family=='all' or v.startswith(args.family+'_')]
    assert variants
    args.output.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(regularization=REG_CONFIG,encoder_training='fully frozen; no stage4 adaptation',variants=variants,models_trained=len(variants)*5,logged_epochs=len(variants)*200,checkpoints_saved=len(variants)*10,
        training_platform='Kaggle T4' if sys.platform!='win32' and Path('/kaggle/working').is_dir() else 'local',
        torch_version=str(torch.__version__),timm_version=timm.__version__,
        gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        train_manifest_sha256=file_hash(args.artifacts/'train.csv'),
        primary_epoch=20,primary_metric='identity',snapshots=[20,40],full_model_ema=.97,
        batchnorm='eval; affine and buffers frozen',test_reference_labels_read=False,competition_submission_performed=False)
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    smoke={}
    for variant in variants:
        args.variant,args.fold,args.device=variant,0,'cuda:0'
        smoke[variant]=worker(args,smoke=True)
        print('GPU PREFLIGHT PASS',variant,flush=True)
    (args.output/'gpu_smoke_checks.json').write_text(json.dumps(smoke,indent=2))
    jobs=[(variant,fold) for fold in range(5) for variant in variants]
    active={}
    while jobs or active:
        for gpu in range(min(2,torch.cuda.device_count())):
            if gpu in active or not jobs: continue
            variant,fold=jobs.pop(0)
            folder=args.output/variant/f'fold{fold}';folder.mkdir(parents=True,exist_ok=True)
            path=folder/'worker.log';log=path.open('w',buffering=1)
            cmd=[sys.executable,'-u',__file__,'--worker','--variant',variant,'--fold',str(fold),'--device',f'cuda:{gpu}',
                '--data',str(args.data),'--assets',str(args.assets),'--artifacts',str(args.artifacts),'--output',str(args.output),'--seed',str(args.seed)]
            active[gpu]=dict(process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log=log,path=path,offset=0)
            print('Launched',variant,'fold',fold,'gpu',gpu,flush=True)
        for gpu,item in list(active.items()):
            with item['path'].open() as f:
                f.seek(item['offset']);new=f.read();item['offset']=f.tell()
            if new: print(new,end='',flush=True)
            code=item['process'].poll()
            if code is not None:
                item['log'].close()
                if code:
                    for other in active.values():
                        if other['process'].poll() is None: other['process'].terminate()
                    raise RuntimeError(f"Worker failed: {item['path']}")
                del active[gpu]
        if active: time.sleep(3)
    print(f'COMPLETE {len(variants)*5} contrast trajectories, {len(variants)*200} logged epochs, {len(variants)*10} snapshots.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,default=Path('artifacts'));p.add_argument('--output',type=Path,default=Path('contrast_run'))
    p.add_argument('--variant',choices=VARIANTS);p.add_argument('--fold',type=int,default=0);p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,default=20261026);p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--schedule-epochs',type=int,default=80);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--encoder-lr',type=float,default=1e-5)
    p.add_argument('--worker',action='store_true');p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--family',choices=['all','hrnet','rfm'],default='all')
    args=p.parse_args()
    assert args.epochs==40 and args.schedule_epochs==80 and args.batch_size==8
    if args.smoke_only:
        args.output.mkdir(parents=True,exist_ok=True)
        print(json.dumps(worker(args,True),indent=2))
    elif args.worker: worker(args)
    else: run(args)
