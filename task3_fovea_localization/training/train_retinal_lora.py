"""Full80 retinal LoRA with disposable update checks and checkpoint replays."""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from retinal_lora_models import SpatialModel,LoRALinear,frozen_hash,adapter_hash,predict
from spatial_models import ImageBank,make_batch,objective,file_hash,state_hash,seed_everything
from next_results import write_submission


def load(path,assets,device):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=SpatialModel(saved['model_name'],assets,saved['seed'],load_pretrained=False)
    model.load_state_dict(saved['state_dict'],strict=True)
    model.to(device).eval()
    assert frozen_hash(model.encoder)==saved['frozen_encoder_sha256']
    assert adapter_hash(model.encoder)!=saved['initial_adapter_sha256']
    return model,saved


def main(args):
    assert torch.cuda.is_available()
    seed_everything(args.seed);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda:0')
    train=pd.read_csv('artifacts/train.csv');test=pd.read_csv('artifacts/test.csv')
    assert len(train)==80 and len(test)==20
    bank=ImageBank(train,args.data,'train',device);tests=ImageBank(test,args.data,'test',device)
    pd.concat([bank.geometry,tests.geometry]).to_csv(args.output/'image_geometry.csv',index=False)
    model=SpatialModel(args.model,args.assets,args.seed).to(device)
    frozen,initial_adapter,initial_head=frozen_hash(model.encoder),adapter_hash(model.encoder),state_hash(model.head)
    files=['train_retinal_lora.py','retinal_lora_models.py','next_models.py','next_results.py','spatial_models.py','fovea.py']
    config=dict(model_name=args.model,seed=args.seed,training_scope='all80',train_ids=train.image_id.tolist(),
        validation_ids=[],oof_available=False,epochs=80,schedule_epochs=80,adapter_rank=8,adapter_alpha=8,
        adapter_blocks=model.encoder.adapter_blocks,adapter_parameters=110592,adapter_lr=.0001,head_lr=.001,
        adapter_warmup_epochs=5,full_model_ema=.97,source_load_info=model.encoder.load_info,
        source_file_sha256=model.encoder.provenance['sha256'],source_sha256={f:file_hash(f) for f in files},
        train_manifest_sha256=file_hash('artifacts/train.csv'),gpu=torch.cuda.get_device_name(device),
        torch_version=str(torch.__version__),test_reference_labels_read_for_training=False,
        reference_is_development_metric=True,competition_submission_performed=False)
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    snapshot=args.output/'executed_source';snapshot.mkdir(exist_ok=True)
    for file in files: shutil.copy2(file,snapshot/file)
    # Temporary model is discarded. Original head, adapters and encoder stay untouched.
    ids=np.arange(8);images,xy,boxes,theta=make_batch(bank,ids,augment=True)
    with torch.inference_mode():
        original=model.encoder(images,amp=False)
        for module in model.encoder.modules():
            if isinstance(module,LoRALinear): module.scale=0
        disabled=model.encoder(images,amp=False)
        for left,right in zip(original,disabled): torch.testing.assert_close(left,right,atol=0,rtol=0)
        for module in model.encoder.modules():
            if isinstance(module,LoRALinear): module.scale=1
    temporary=copy.deepcopy(model).train();temporary.encoder.adapter_active=True
    parameters=[p for p in temporary.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=.001)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024,growth_interval=2000)
    loss=objective(temporary(images,amp=True),xy,boxes,theta,bank,ids)
    assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(optimizer)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    assert all(p.grad is None for p in temporary.parameters() if not p.requires_grad)
    adapter_grad=sum(float(p.grad.abs().sum()) for name,p in temporary.encoder.named_parameters() if 'lora_' in name)
    assert adapter_grad>0
    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(parameters,5))
    scaler.step(optimizer);scaler.update()
    assert adapter_hash(temporary.encoder)!=initial_adapter and state_hash(temporary.head)!=initial_head
    assert frozen_hash(temporary.encoder)==frozen and frozen_hash(model.encoder)==frozen
    assert adapter_hash(model.encoder)==initial_adapter and state_hash(model.head)==initial_head
    (args.output/'gpu_preflight.json').write_text(json.dumps(dict(status='PASS',batch_size=8,
        zero_adapter_initial_function_exact=True,actual_adapter_update=True,actual_head_update=True,
        original_pretrained_parameters_unchanged=True,formal_model_unchanged=True,
        adapter_gradient_sum=adapter_grad,loss=float(loss.detach())),indent=2))
    del temporary,parameters,optimizer,scaler,loss,original,disabled,images
    gc.collect();torch.cuda.empty_cache()
    seed_everything(args.seed);torch.set_num_threads(2)
    adapters=[p for name,p in model.encoder.named_parameters() if 'lora_' in name]
    optimizer=torch.optim.AdamW([dict(params=model.head.parameters(),lr=.001),dict(params=adapters,lr=.0001)],weight_decay=1e-4)
    trainable={name:p for name,p in model.named_parameters() if p.requires_grad}
    scaler=torch.amp.GradScaler('cuda',init_scale=1024,growth_interval=2000)
    ema=copy.deepcopy(model).eval().requires_grad_(False);ema_parameters=dict(ema.named_parameters())
    history=[];start=time.perf_counter()
    print('START LoRA full80',args.model,'adapter_parameters=110592; no OOF',flush=True)
    for epoch in range(1,81):
        model.train();model.encoder.adapter_active=epoch>5
        factor=min(1,epoch/5)*(.05+.95*.5*(1+math.cos(math.pi*max(0,epoch-5)/75)))
        optimizer.param_groups[0]['lr']=.001*factor
        optimizer.param_groups[1]['lr']=.0001*factor if epoch>5 else 0
        losses=[];norms=[];skipped=0
        for ids in np.array_split(np.random.permutation(80),10):
            images,xy,boxes,theta=make_batch(bank,ids,augment=True)
            optimizer.zero_grad(set_to_none=True)
            loss=objective(model(images,amp=True),xy,boxes,theta,bank,ids)
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(trainable.values(),5)
            oldscale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            skip=scaler.get_scale()<oldscale
            if skip: skipped+=1
            else:
                with torch.no_grad():
                    for name,value in trainable.items(): ema_parameters[name].lerp_(value.detach(),.03)
            losses.append(float(loss.detach()))
            if torch.isfinite(norm): norms.append(float(norm))
        assert norms and skipped<10
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),gradient_norm=float(np.mean(norms)),
            head_lr=optimizer.param_groups[0]['lr'],adapter_lr=optimizer.param_groups[1]['lr'],
            skipped_steps=skipped,elapsed_seconds=time.perf_counter()-start))
        pd.DataFrame(history).to_csv(args.output/'history.csv',index=False)
        if epoch==1 or epoch%10==0: print(f'{args.model} LoRA epoch={epoch}/80 loss={np.mean(losses):.4f} skipped={skipped}',flush=True)
        if epoch in [20,40,80]:
            assert frozen_hash(model.encoder)==frozen and frozen_hash(ema.encoder)==frozen
            dest=args.output/f'epoch{epoch}';dest.mkdir(exist_ok=True)
            saved=dict(model_name=args.model,seed=args.seed,completed_epochs=epoch,
                state_dict={key:value.detach().cpu().clone() for key,value in ema.state_dict().items()},
                config=config,frozen_encoder_sha256=frozen,initial_adapter_sha256=initial_adapter,
                initial_head_sha256=initial_head)
            torch.save(saved,dest/'model.pt')
            manifest={key:value for key,value in saved.items() if key!='state_dict'}
            manifest.update(checkpoint_sha256=file_hash(dest/'model.pt'),original_pretrained_parameters_unchanged=True,
                actual_adapter_update=adapter_hash(ema.encoder)!=initial_adapter,actual_head_update=state_hash(ema.head)!=initial_head)
            assert manifest['actual_adapter_update'] and manifest['actual_head_update']
            (dest/'training_manifest.json').write_text(json.dumps(manifest,indent=2))
            del saved
    del model,ema,optimizer,scaler,trainable,ema_parameters,adapters,images,loss
    gc.collect();torch.cuda.empty_cache()
    for epoch in [20,40,80]:
        dest=args.output/f'epoch{epoch}';model,saved=load(dest/'model.pt',args.assets,device)
        identity,tta=predict(model,tests,np.arange(20))
        frame=test.copy();frame[['identity_x','identity_y']]=identity;frame[['pred_x','pred_y']]=tta
        frame.to_csv(dest/'test_predictions.csv',index=False)
        write_submission(args.data/'sample_submission.csv',frame,dest/f'submission_lora_{args.model}_e{epoch}.csv',True)
        write_submission(args.data/'sample_submission.csv',frame,dest/f'submission_lora_{args.model}_e{epoch}_tta.csv')
        del model,saved;gc.collect();torch.cuda.empty_cache()
    checks=[]
    for epoch in [20,40,80]:
        dest=args.output/f'epoch{epoch}';model,saved=load(dest/'model.pt',args.assets,device)
        identity,tta=predict(model,tests,np.arange(20))
        frame=pd.read_csv(dest/'test_predictions.csv')
        np.testing.assert_allclose(frame[['identity_x','identity_y']],identity,atol=.003,rtol=0)
        np.testing.assert_allclose(frame[['pred_x','pred_y']],tta,atol=.003,rtol=0)
        manifest=json.loads((dest/'training_manifest.json').read_text())
        assert manifest['checkpoint_sha256']==file_hash(dest/'model.pt')
        checks.append(dict(epoch=epoch,checkpoint_sha256=manifest['checkpoint_sha256'],strict_replay=True))
        del model,saved;gc.collect();torch.cuda.empty_cache()
    for path,sha in config['source_sha256'].items(): assert file_hash(path)==sha and file_hash(snapshot/path)==sha
    (args.output/'verification.json').write_text(json.dumps(dict(status='PASS',models_trained=1,logged_epochs=80,
        checkpoints_replayed=3,training_scope='all80',oof_available=False,checks=checks,
        reference_labels_used_for_training=False,competition_submission_performed=False),indent=2))
    print('COMPLETE full80 LoRA',args.model,'3 strict checkpoint replays',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['rfm','visionfm'],required=True)
    p.add_argument('--seed',type=int,default=20261022);p.add_argument('--data',type=Path,default=Path('data/detection'))
    p.add_argument('--assets',type=Path,default=Path('kaggle_dataset_next_assets'));p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
