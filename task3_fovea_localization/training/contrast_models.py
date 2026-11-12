"""Frozen HRNet/RFM with paired RGB-luminance and green contrast inputs."""
import numpy as np
import torch
from torch import nn

import next_models
import spatial_models
from contrast_image_bank import ContrastImageBank as ImageBank
from spatial_models import file_hash,state_hash,seed_everything,make_batch,objective,score
from next_models import predict

VARIANTS=('hrnet_luma','hrnet_green','rfm_luma','rfm_green')
REG_CONFIG={name:dict(model=name.split('_')[0],contrast=name.split('_')[1],tiles=8,clip_limit=2.,sigma=2)
            for name in VARIANTS}


class AdaptModel(nn.Module):
    def __init__(self,variant,assets,seed=20261026,load_pretrained=True):
        super().__init__();assert variant in VARIANTS
        self.variant=variant
        name=REG_CONFIG[variant]['model']
        api=spatial_models if name=='hrnet' else next_models
        model=api.SpatialModel(name,assets,seed,load_pretrained)
        self.encoder,self.head=model.encoder,model.head

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self

    def set_epoch(self,epoch):
        pass

    def forward(self,images,amp=False):
        return self.head(self.encoder(images,amp))


def frozen_hash(model):
    return state_hash(model.encoder)


def trainable_parameters(model):
    return {name:p for name,p in model.named_parameters() if p.requires_grad}


def load_saved(path,assets,device):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=AdaptModel(saved['variant'],assets,saved['seed'],load_pretrained=False)
    model.load_state_dict(saved['state_dict'],strict=True);model.to(device).eval()
    assert frozen_hash(model)==saved['frozen_state_sha256']
    assert state_hash(model.head)!=saved['initial_head_sha256']
    return model,saved
