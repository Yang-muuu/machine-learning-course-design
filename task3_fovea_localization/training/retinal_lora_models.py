"""Small attention adapters on strictly loaded frozen retinal encoders."""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from next_models import Encoder as FrozenEncoder, predict
from spatial_models import SpatialHead


class LoRALinear(nn.Module):
    def __init__(self,base,rank=8):
        super().__init__()
        assert isinstance(base,nn.Linear)
        self.base=base.requires_grad_(False)
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
        self.scale=1.  # alpha=rank=8
        self.in_features,self.out_features=base.in_features,base.out_features

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self,x):
        return self.base(x)+self.scale*F.linear(F.linear(x,self.lora_A),self.lora_B)


def tensor_hash(items):
    h=hashlib.sha256()
    for key,value in sorted(items):
        value=value.detach().cpu().contiguous()
        h.update((key+str(value.dtype)+str(tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def frozen_hash(encoder):
    return tensor_hash((key,value) for key,value in encoder.state_dict().items() if 'lora_' not in key)


def adapter_hash(encoder):
    return tensor_hash((key,value) for key,value in encoder.state_dict().items() if 'lora_' in key)


class Encoder(FrozenEncoder):
    def __init__(self,name,assets,load_pretrained=True):
        assert name in ['rfm','visionfm']
        super().__init__(name,assets,load_pretrained)
        self.adapter_active=False
        self.adapter_blocks=list(range(len(self.base.blocks)-3,len(self.base.blocks)))
        for index in self.adapter_blocks:
            attention=self.base.blocks[index].attn
            attention.qkv=LoRALinear(attention.qkv)
            attention.proj=LoRALinear(attention.proj)
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==110592

    def forward(self,images,amp=False):
        gradients=bool(self.adapter_active and amp and torch.is_grad_enabled())
        with torch.set_grad_enabled(gradients):
            x=(images-self.mean)/self.std
            with torch.autocast(device_type=x.device.type,dtype=torch.float16,enabled=amp and x.is_cuda):
                if self.name=='rfm':
                    features=self.base.forward_intermediates(x,indices=[1,3,5,8],
                        intermediates_only=True,output_fmt='NCHW',norm=False,stop_early=True)
                else:
                    features=self.base.get_intermediate_layers(x,n=[2,5,8,11],reshape=True,norm=True)
            assert len(features)==4
            return [value.float() if gradients else value.detach().float() for value in features]


class SpatialModel(nn.Module):
    def __init__(self,name,assets,seed,load_pretrained=True):
        super().__init__()
        self.encoder=Encoder(name,assets,load_pretrained)
        torch.manual_seed(seed)
        self.head=SpatialHead(self.encoder.channels)

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self

    def forward(self,images,amp=False):
        return self.head(self.encoder(images,amp))
