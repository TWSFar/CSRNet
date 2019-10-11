import torch
from torchvision import models


mod = models.vgg16(pretrained=False)
mm = torch.load("/home/twsf/.cache/torch/checkpoints/vgg16-397923af.pth")
iterm = mod.state_dict()
iterm2 = iterm.items()
for i in range(len(iterm2)):
    p = mod.state_dict()
    p2 = p1[1]
    p3 = p2.data
pass
