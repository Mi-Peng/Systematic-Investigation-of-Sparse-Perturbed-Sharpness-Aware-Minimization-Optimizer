from data.build import build_dataset, build_train_dataloader, build_val_dataloader
from models.build import build_model
from solver.build import build_optimizer
import ipdb



# from configs.defaulf_cfg import default_parser
# cfg_file = default_parser()
# args = cfg_file.get_args()
# train_data, val_data, n_classes = build_dataset(args)
# args.n_classes = n_classes
# model = build_model(args)
# optimizer, base_optimizer = build_optimizer(args, model=model)
# optimizer.update_mask(model, train_data=train_data)
# ipdb.set_trace()

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
# https://blog.csdn.net/ViatorSun/article/details/119940759

import torch.autograd as autograd

class MyConv2d_Lay(autograd.Function):
    @staticmethod
    def forward(ctx, weight, inp_unf):
        ctx.save_for_backward(weight, inp_unf)
        w_s = weight 
        out_unf = inp_unf.matmul(w_s)
        return out_unf

    @staticmethod
    def backward(ctx, g):
        weight, inp_unf = ctx.saved_tensors
        w_s = (weight).t()

        g_w_s = inp_unf.transpose(1,2).matmul(g).sum(0)
        g_inp_unf = g.matmul(w_s)
        return g_w_s , g_inp_unf, None
    
class NMConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flag = False
        self.permute_idx = [v for v in range(self.weight.view(self.weight.size(0), -1).size(0))]
        self.forward_mask = torch.zeros(self.weight.view(self.weight.size(0), -1).t().shape).requires_grad_(False)
        self.backward_mask = torch.zeros(self.weight.view(self.weight.size(0), -1).t().shape).requires_grad_(False)
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)


    def forward(self, x):
        w = self.weight.view(self.weight.size(0), -1).t()   
        inp_unf = self.unfold(x)
        out_unf = MyConv2d_Lay.apply(w, inp_unf.transpose(1, 2))
        if self.flag == False:
            self.fold = nn.Fold((int(math.sqrt(out_unf.shape[1])), int(math.sqrt(out_unf.shape[1]))), (1,1))
            self.flag = True
        out = self.fold(out_unf.transpose(1, 2))
        return out

# x = torch.randn(1, 1, 3, 3)
# cnn = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1)

# print(cnn.forward(x))

from models.resnet import resnet18
from solver.utils import get_nm_mask, get_fisher
import torchvision
import numpy as np
from models.nm_conv import SAMConv

mean = np.array([125.3, 123.0, 113.9]) / 255.0
std = np.array([63.0, 62.1, 66.7]) / 255.0
train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
            # Cutout()
        ])
train_data = torchvision.datasets.CIFAR10(root='../cifar', train=True, transform=train_transform, download=True)

model = resnet18(num_classes=10, samconv=True).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

fisher_dict = get_fisher(model, train_data, 12, optimizer.param_groups)

structured_fisher = {}
p_num = {}
for group in optimizer.param_groups:
    for p in group['params']:
        fisher = fisher_dict[id(p)]
        fisher = torch.mean(fisher).item()
        structured_fisher[id(p)] = fisher
        p_num[id(p)] = p.numel()

result = sorted(structured_fisher.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)
for idp, fisher in result:
    pass


import ipdb; ipdb.set_trace()
total_p = sum(list(p_num.values()))
fisher_list = torch.tensor(list(structured_fisher.values()))
_value, _index = torch.sort(fisher_list, descending=True)
import ipdb; ipdb.set_trace()

# set mask
mask_dict = {k: 0 for k in fisher_dict.keys()}




# cnn = NMConv(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1)
# ##################### main part ###########################
# unfold = nn.Unfold(kernel_size=3, padding=1, stride=1, dilation=cnn.dilation)
# w = cnn.weight.view(cnn.weight.size(0), -1).t()
# inp_unf = unfold(x)
# out_unf = inp_unf.transpose(1,2).matmul(w)
# fold = nn.Fold((int(math.sqrt(out_unf.shape[1])), int(math.sqrt(out_unf.shape[1]))), (1,1))
# out = fold(out_unf.transpose(1,2))
# print(out)
# ##################### main part ###########################
# out = cnn(x)
# print(out)
# out = F.conv2d(x, cnn.weight, padding=1, stride=1)
# print(out)