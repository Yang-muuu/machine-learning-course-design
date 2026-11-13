"""Refit a selected pretrained model on all 80 training labels; no OOF claim."""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import spatial_models
import next_models
from spatial_models import (ImageBank,make_batch,objective,file_hash,state_hash,seed_everything,score)
from next_results import write_submission


def api_for(name):
    return spatial_models if name in ['hrnet','green','dinov2'] else next_models


def load(path,assets,device):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=api_for(saved['model_name']).SpatialModel(saved['model_name'],assets,saved['seed'],load_pretrained=False)
    model.load_state_dict(saved['state_dict'],strict=True);model.to(device).eval()
    assert state_hash(model.encoder)==saved['frozen_encoder_sha256']
    return model,saved


def run(args):
    assert torch.cuda.is_available()
    seed_everything(args.seed);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device);torch.cuda.set_device(device)
    train=pd.read_csv(args.artifacts/'train.csv');test=pd.read_csv(args.artifacts/'test.csv')
    assert len(train)==80 and len(test)==20 and train.image_id.is_unique
    bank=ImageBank(train,args.data,'train',device);test_bank=ImageBank(test,args.data,'test',device)
    model=api_for(args.model).SpatialModel(args.model,args.assets,args.seed).to(device)
    frozen=state_hash(model.encoder);head_initial=state_hash(model.head)
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=.001,weight_decay=1e-4)
    ema=copy.deepcopy(model.head).eval().requires_grad_(False)
    seed_everything(args.seed);torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=True)
    config=dict(model_name=args.model,seed=args.seed,epochs=80,schedule_epochs=80,training_scope='all_80',
        training_image_ids=train.image_id.tolist(),validation_image_ids=[],oof_available=False,
        torch_version=str(torch.__version__),gpu=torch.cuda.get_device_name(device),
        train_manifest_sha256=file_hash(args.artifacts/'train.csv'),
        source_file_sha256=model.encoder.provenance['sha256'],source_load_info=model.encoder.load_info,
        training_reference_labels_read=False,competition_submission_performed=False)
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    pd.concat([bank.geometry,test_bank.geometry]).to_csv(args.output/'image_geometry.csv',index=False)
    history=[];started=time.perf_counter()
    ids=np.arange(80)
    print('START full80',args.model,'seed',args.seed,'no OOF or validation split',flush=True)
    for epoch in range(1,81):
        model.train()
        factor=min(1,epoch/5)*(.05+.95*.5*(1+math.cos(math.pi*max(0,epoch-5)/75)))
        optimizer.param_groups[0]['lr']=.001*factor
        losses,norms=[],[]
        for indices in np.array_split(np.random.permutation(ids),10):
            images,xy,boxes,theta=make_batch(bank,indices,augment=True)
            optimizer.zero_grad(set_to_none=True)
            loss=objective(model(images,amp=True),xy,boxes,theta,bank,indices)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None for p in model.encoder.parameters())
            norm=torch.nn.utils.clip_grad_norm_(model.head.parameters(),5)
            assert torch.isfinite(norm)
            optimizer.step()
            with torch.no_grad():
                for average,current in zip(ema.parameters(),model.head.parameters()): average.lerp_(current.detach(),.03)
            losses.append(float(loss.detach()));norms.append(float(norm))
        row=dict(epoch=epoch,loss=float(np.mean(losses)),gradient_norm=float(np.mean(norms)),lr=.001*factor,
                 elapsed_seconds=time.perf_counter()-started)
        if epoch==1 or epoch%10==0:
            identity,tta=api_for(args.model).predict(model,bank,ids,head=ema)
            row['training_fit_rmse']=score(train[['x','y']],identity)['rmse_pixels']
            print(f"{args.model} full80 epoch={epoch}/80 loss={row['loss']:.4f} TRAIN_fit={row['training_fit_rmse']:.3f} (not OOF)",flush=True)
        history.append(row);pd.DataFrame(history).to_csv(args.output/'history.csv',index=False)
        if epoch in [20,40,80]:
            assert state_hash(model.encoder)==frozen
            dest=args.output/f'epoch{epoch}';dest.mkdir(exist_ok=True)
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            for key,value in ema.state_dict().items(): state['head.'+key]=value.detach().cpu().clone()
            checkpoint=dict(model_name=args.model,seed=args.seed,completed_epochs=epoch,schedule_epochs=80,
                state_dict=state,frozen_encoder_sha256=frozen,initial_head_sha256=head_initial,
                source_file_sha256=config['source_file_sha256'],source_load_info=config['source_load_info'],
                train_ids=train.image_id.tolist(),validation_ids=[],training_scope='all_80',oof_available=False)
            torch.save(checkpoint,dest/'model.pt')
            manifest={k:v for k,v in checkpoint.items() if k!='state_dict'}
            manifest.update(checkpoint_sha256=file_hash(dest/'model.pt'),frozen_encoder_unchanged=True,
                actual_head_update=state_hash(ema)!=head_initial,checkpoint_selection='predeclared20/40/80 full-data EMA snapshots')
            (dest/'training_manifest.json').write_text(json.dumps(manifest,indent=2))
            del state,checkpoint
    del model,ema,optimizer
    gc.collect();torch.cuda.empty_cache()
    # Every prediction comes from a saved checkpoint, then gets independently replayed once more.
    for epoch in [20,40,80]:
        dest=args.output/f'epoch{epoch}'
        model,saved=load(dest/'model.pt',args.assets,device)
        identity,tta=api_for(args.model).predict(model,test_bank,np.arange(20))
        table=test.copy();table[['identity_x','identity_y']]=identity;table[['pred_x','pred_y']]=tta
        table.to_csv(dest/'test_predictions.csv',index=False)
        write_submission(args.data/'sample_submission.csv',table,dest/f'submission_full_{args.model}_e{epoch}.csv',True)
        write_submission(args.data/'sample_submission.csv',table,dest/f'submission_full_{args.model}_e{epoch}_tta.csv')
        del model,saved
        gc.collect();torch.cuda.empty_cache()
    checks=[]
    for epoch in [20,40,80]:
        dest=args.output/f'epoch{epoch}'
        manifest=json.loads((dest/'training_manifest.json').read_text())
        assert manifest['checkpoint_sha256']==file_hash(dest/'model.pt')
        model,saved=load(dest/'model.pt',args.assets,device)
        assert saved['train_ids']==train.image_id.tolist() and saved['validation_ids']==[]
        assert state_hash(model.head)!=saved['initial_head_sha256']
        identity,tta=api_for(args.model).predict(model,test_bank,np.arange(20))
        table=pd.read_csv(dest/'test_predictions.csv')
        np.testing.assert_allclose(table[['identity_x','identity_y']],identity,atol=.003,rtol=0)
        np.testing.assert_allclose(table[['pred_x','pred_y']],tta,atol=.003,rtol=0)
        checks.append(dict(epoch=epoch,checkpoint_sha256=manifest['checkpoint_sha256'],strict_replay=True))
        del model,saved
        gc.collect();torch.cuda.empty_cache()
    verification=dict(status='PASS',models_trained=1,training_images=80,logged_epochs=80,checkpoints_replayed=3,
        oof_available=False,reference_labels_read_for_training=False,competition_submission_performed=False,checks=checks)
    (args.output/'verification.json').write_text(json.dumps(verification,indent=2))
    print('COMPLETE full-data',args.model,'3 strict-replayed snapshots, no OOF claim',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['hrnet','green','rfm','visionfm','flair','dinov3_convnext'],required=True)
    p.add_argument('--seed',type=int,default=20261022);p.add_argument('--device',default='cuda:0')
    p.add_argument('--data',type=Path,default=Path('data/detection'));p.add_argument('--artifacts',type=Path,default=Path('artifacts'))
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
