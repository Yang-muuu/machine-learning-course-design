"""2D per-image soft-clDice, adapted from the authors' MIT-licensed reference.

Original: Copyright (c) 2021 Johannes C. Paetzold and Suprosanna Shit.
https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
See CLDICE_LICENSE.txt. This version honors the requested iteration count and
averages image scores rather than aggregating all batch pixels together.
"""
import torch
from torch.nn import functional as F


def erode(image):
    vertical = -F.max_pool2d(-image, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-image, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_skeleton(image, iterations=8):
    opened = F.max_pool2d(erode(image), 3, stride=1, padding=1)
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = erode(image)
        opened = F.max_pool2d(erode(image), 3, stride=1, padding=1)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def cldice_loss(probability, target, iterations=8):
    probability, target = probability.float(), target.float()
    predicted_skeleton = soft_skeleton(probability, iterations)
    with torch.no_grad():
        target_skeleton = soft_skeleton(target, iterations)
    axes = (1, 2, 3)
    precision = ((predicted_skeleton * target).sum(axes) + 1) / (predicted_skeleton.sum(axes) + 1)
    sensitivity = ((target_skeleton * probability).sum(axes) + 1) / (target_skeleton.sum(axes) + 1)
    return (1 - 2 * precision * sensitivity / (precision + sensitivity).clamp_min(1e-7)).mean()
