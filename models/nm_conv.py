import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import math
from solver.utils import get_fisher, get_nm_mask

class MaskedMatMulConv(Function):
    @staticmethod
    def forward(ctx, w_matrix, input_unfold, bpmask):
        """
        w_matrix: (c_in*k*k, c_o)
        input_unfold: (b, hw, c_in*k*k)
        """
        ctx.save_for_backward(w_matrix, input_unfold, bpmask)

        output_unfold = input_unfold.matmul(w_matrix) # (b, hw, c_in*k*k) * (c_in*k*k, c_o) -> (b, hw, c_o)
        
        return output_unfold

    @staticmethod
    def backward(ctx, g_o):
        w_matrix, input_unfold, bpmask = ctx.saved_tensors
        if bpmask is not None:
            w_matrix = (w_matrix * bpmask).t()
        else:
            w_matrix = w_matrix.t()
        g_input = g_o.matmul(w_matrix) # (hw, co) * (co, c_in*k*k) -> (hw, c_in*k*k)
        g_w = input_unfold.transpose(1,2).matmul(g_o).sum(0) # (b, c_in*k*k, hw) * (hw, co) -> (b,c_in*k*k, hw) -> (c_in*k*k, hw)

        return g_w, g_input, None
    

class MatMulConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flag = False
        self.bpmask = None
        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size, 
            dilation=self.dilation, 
            padding=self.padding, 
            stride=self.stride
        )
    def forward(self, x, bpmask=None):
        w_matrix = self.weight.view(self.weight.size(0), -1).t() # (c_o, c_in, k, k) -> (c_o, kkc_i) -> (kkc_i, c_o)
        input_unfold = self.unfold(x) # (b, c_in, h, w) -> (b, c_in*k*k, hw)
        bpmask = bpmask.view(bpmask.size(0), -1).t()
        output_unfold = MaskedMatMulConv.apply(w_matrix, input_unfold.transpose(1,2), bpmask)

        if self.flag == False:
            self.fold = nn.Fold((int(math.sqrt(output_unfold.shape[1])), int(math.sqrt(output_unfold.shape[1]))), (1,1))
            self.flag = True
        output = self.fold(output_unfold.transpose(1,2)) # (b, hw, c_o) -> (b, c_o, hw) -> (b, c_o, h, w)
        return output
    
    def cnn_forward(self, x):
        return super().forward(x)

class SAMConv(MatMulConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin = True
        self.mask = torch.ones_like(self.weight, requires_grad=False)

    def set_origin(self, bool_var):
        self.origin = bool_var

    def set_bpmask(self, mask):
        self.mask = mask
        self.mask.requires_grad = False

    def forward(self, x):
        if self.origin:
            return super().cnn_forward(x)
        else:
            return super().forward(x, bpmask=self.mask)

def set_conv_model_origin(model, bool):
    for m in model.modules():
        if isinstance(m, SAMConv):
            m.set_origin(bool)


def update_conv_inter_bpmask(model, train_data, num_samples, param_groups, pattern, n, m, sparsity):
    assert pattern in ['structured', 'nm']
    fisher_dict = get_fisher(model, train_data, num_samples, param_groups)

    if pattern == 'structured':
        structured_fisher = {}
        p_num = {}
        for group in param_groups:
            for p in group['params']:
                fisher = fisher_dict[id(p)]
                fisher = torch.mean(fisher)
                structured_fisher[id(p)] = fisher
                p_num[id(p)] = p.numel()
        
        now_p, total_p = 0, sum(list(p_num.values()))
        mask_dict = {k: 0 for k in fisher_dict.keys()}
        result = sorted(structured_fisher.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)
        
        for idp, fisher in result:
            if now_p < int(total_p * (1 - sparsity)):
                mask_dict[idp] = 1
                now_p += p_num[idp]
            else:
                break
        
        # set mask into module
        module_list = []
        for module in model.modules():
            if isinstance(module, SAMConv):
                module_list.append(module)
        if len(module_list) == 0: raise ValueError
        for module in module_list:
            fisher = fisher_dict[id(module.weight)]
            mask = torch.ones_like(module.weight) * mask_dict[id(module.weight)]
            module.set_bpmask(mask)
    elif pattern == 'nm':
        module_list = []
        for module in model.modules():
            if isinstance(module, SAMConv):
                module_list.append(module)
        if len(module_list) == 0: raise ValueError
        for module in module_list:
            fisher = fisher_dict[id(module.weight)]
            mask = get_nm_mask(fisher, n, m)
            module.set_bpmask(mask)