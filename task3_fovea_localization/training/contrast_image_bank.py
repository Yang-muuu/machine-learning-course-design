"""Fixed tile-wise contrast equalization without changing image geometry."""
import numpy as np
from PIL import Image
import torch

from spatial_models import ImageBank as BaseImageBank


def clahe_u8(channel,tiles=8,clip_limit=2.):
    height,width=channel.shape
    assert channel.dtype==np.uint8 and height%tiles==width%tiles==0
    th,tw=height//tiles,width//tiles
    maps=np.empty((tiles,tiles,256),dtype=np.float32)
    limit=max(1,int(clip_limit*th*tw/256))
    for ty in range(tiles):
        for tx in range(tiles):
            hist=np.bincount(channel[ty*th:(ty+1)*th,tx*tw:(tx+1)*tw].ravel(),minlength=256)
            excess=int(np.maximum(hist-limit,0).sum());hist=np.minimum(hist,limit)
            hist+=excess//256
            remainder=excess%256
            if remainder: hist[(np.arange(remainder)*256//remainder).astype(int)]+=1
            assert hist.sum()==th*tw
            maps[ty,tx]=np.cumsum(hist)*255/(th*tw)
    yy=np.arange(height)/th-.5;xx=np.arange(width)/tw-.5
    y0=np.floor(yy).astype(int);x0=np.floor(xx).astype(int)
    wy=(yy-y0)[:,None];wx=(xx-x0)[None,:]
    y1=np.clip(y0+1,0,tiles-1);x1=np.clip(x0+1,0,tiles-1)
    y0=np.clip(y0,0,tiles-1);x0=np.clip(x0,0,tiles-1)
    a=maps[y0[:,None],x0[None,:],channel]
    b=maps[y0[:,None],x1[None,:],channel]
    c=maps[y1[:,None],x0[None,:],channel]
    d=maps[y1[:,None],x1[None,:],channel]
    return np.clip(np.rint((a*(1-wx)+b*wx)*(1-wy)+(c*(1-wx)+d*wx)*wy),0,255).astype(np.uint8)


def transform_rgb(rgb,variant):
    assert rgb.dtype==np.uint8 and rgb.shape==(448,448,3)
    black=rgb.max(axis=2)<13
    if variant=='luma':
        ycbcr=np.asarray(Image.fromarray(rgb).convert('YCbCr')).copy()
        ycbcr[:,:,0]=clahe_u8(ycbcr[:,:,0])
        result=np.asarray(Image.fromarray(ycbcr,mode='YCbCr').convert('RGB')).copy()
    elif variant=='green':
        green=clahe_u8(rgb[:,:,1])
        result=np.repeat(green[:,:,None],3,axis=2)
    else: raise ValueError(variant)
    result[black]=0
    assert result.shape==rgb.shape and result.dtype==np.uint8
    return result


class ContrastImageBank(BaseImageBank):
    variant='luma'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        original=self.images.cpu().numpy().transpose(0,2,3,1)
        transformed=np.stack([transform_rgb(image,self.variant) for image in original])
        self.images=torch.from_numpy(transformed.transpose(0,3,1,2).copy()).to(self.images.device)
