"""AMP兼容的两步SAM；一次混合标签、冻结BN、精确恢复扰动、一次AdamW和EMA更新。"""
import torch
from cnn_models import single_loss,update_ema
from model_and_aug import mix_batch


def forward_loss(model,images,first,second,ratio,config,epoch,device):
    with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
        logits=model(images)
        supervised=ratio*single_loss(logits,first,config,epoch)+(1-ratio)*single_loss(logits,second,config,epoch)
        auxiliary=model.pop_auxiliary_loss()
        loss=supervised+(auxiliary if auxiliary is not None else 0.)
    assert torch.isfinite(loss), 'SAM损失出现NaN/Inf'
    return logits,supervised,auxiliary,loss


def fields(model,supervised,auxiliary,first_loss,attack):
    model.last_supervised_loss=float(supervised.detach())
    model.last_auxiliary_loss=float(auxiliary.detach()) if auxiliary is not None else 0.
    model.last_consistency_loss=0.
    model.last_sam_first_loss=float(first_loss.detach())
    model.last_sam_ascent_norm=float(attack)


def train_step_sam(model,ema,images,labels,optimizer,scaler,config,device,epoch,forced=None):
    assert config.sam_rho>0 and config.freeze_bn and config.rdrop_weight==0
    images=images.to(device,non_blocking=True,memory_format=torch.channels_last)
    labels=labels.to(device,non_blocking=True)
    if epoch<=config.epochs-config.unmixed_final_epochs:
        images,first,second,ratio,kind=mix_batch(images,labels,config,forced=forced)
    else:
        first,second,ratio,kind=labels,labels,1.,'none'
    optimizer.zero_grad(set_to_none=True)
    logits,supervised,auxiliary,loss=forward_loss(model,images,first,second,ratio,config,epoch,device)
    scaler.scale(loss).backward()
    parameters=[parameter for group in optimizer.param_groups for parameter in group['params'] if parameter.grad is not None]
    assert parameters, 'SAM没有可用梯度'
    # 第一遍只做扰动，不调用unscale_，避免同一GradScaler对一次优化器更新重复unscale。
    scale=float(scaler.get_scale())
    finite=all(bool(torch.isfinite(parameter.grad).all()) for parameter in parameters)
    if not finite:
        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.update(new_scale=scale*.5)
        model.last_optimizer_step_skipped=True
        fields(model,supervised,auxiliary,loss,0.)
        correct=ratio*(logits.detach().argmax(1)==first).float().sum()+(1-ratio)*(logits.detach().argmax(1)==second).float().sum()
        return float(loss.detach()),float(correct),kind
    with torch.no_grad():
        for parameter in parameters:
            parameter.grad.div_(scale)
        norm=torch.linalg.vector_norm(torch.stack([parameter.grad.float().norm() for parameter in parameters])).clamp_min(1e-12)
        old=[parameter.detach().clone() for parameter in parameters]
        squared=[]
        for parameter in parameters:
            perturbation=parameter.grad*(config.sam_rho/norm)
            parameter.add_(perturbation)
            squared.append(perturbation.float().square().sum())
        attack=torch.stack(squared).sum().sqrt()
    optimizer.zero_grad(set_to_none=True)
    try:
        # 两遍共享同一次MixUp/CutMix，第二遍的Dropout按训练模式独立采样。
        perturbed_logits,second_supervised,second_auxiliary,perturbed_loss=forward_loss(
            model,images,first,second,ratio,config,epoch,device)
        scaler.scale(perturbed_loss).backward()
    finally:
        # copy_精确恢复，避免反复加减扰动产生浮点漂移；异常时也不能留下扰动权重。
        with torch.no_grad():
            for parameter,value in zip(parameters,old):
                parameter.copy_(value)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False)
    previous=float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    model.last_optimizer_step_skipped=scaler.get_scale()<previous
    if ema is not None and not model.last_optimizer_step_skipped:
        update_ema(ema,model,0. if epoch<=config.ema_warmup_epochs else config.ema_decay)
    fields(model,second_supervised,second_auxiliary,loss,attack)
    predicted=perturbed_logits.detach().argmax(1)
    correct=ratio*(predicted==first).float().sum()+(1-ratio)*(predicted==second).float().sum()
    return float(perturbed_loss.detach()),float(correct),kind
