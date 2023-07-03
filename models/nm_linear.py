import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import spmm
from solver.utils import get_fisher, get_nm_mask

class Matmul_NMBP(Function):
    @staticmethod
    def forward(ctx, index, weight, input, bias, weight_t_mask):
        """
        weight: (out_features, in_features)
        bias: (out_features)
        input: (bsz, (n, in_features))
        output: (bsz, (n, out_features))

        weight_t_mask: (in_features, out_features)
        """
        ctx.index = index
        ctx.save_for_backward(weight, input, bias, weight_t_mask)
        output = F.linear(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, g_o):
        index = ctx.index
        weight, input, bias, weight_t_mask = ctx.saved_tensors
        """
            Apply 2:4 cusparseLt matrix multiplication
        g_o: (bsz, (n, out_features))
        weight_t_mask: (in_features, out_features)
        """
        # g_input(bsz, (n, in_features)) 
        #     = g_o(bsz, (n, out_features)) \times weight (out_features, in_features)
        #     = (weight.t() \times g_o.t()).t()
        #     = (weight.t(), (in_features, out_features) \times g_o.t()(bsz, (out_features, n)).t()

        g_input = torch.zeros_like(input)

        # spmm.compressMatrix(index, weight_t_mask)

        spmm.spmm(index, g_o.transpose(-1, -2).contiguous(), g_input.transpose(-1, -2).contiguous())

        # g_weight: = g_o.t() \times input
        g_w = torch.bmm(g_o.transpose(-1, -2), input)
        g_w = torch.sum(g_w, 0)

        # g_bias:
        if bias is not None:
            g_b = torch.sum(torch.sum(g_o, 0), 0)
        else:
            g_b = None
        return None, g_w, g_input, g_b, None


class NMBP_Linear(nn.Linear):
    def __init__(self, 
        index, bs, 
        in_features, out_features, input_rows, 
        bias=True
    ):
        super().__init__(in_features, out_features, bias)
        """
            self.weight is a matrix with size of [out_features, in_features]
            so its forward equals to "torch.mm(self.weight.t(), x) + self.bias"
        """
        assert isinstance(index, int)
        self.index = index
        self.origin = True
        self.mask = torch.ones_like(self.weight, requires_grad=False)

        self.inf = in_features
        self.outf = out_features

        self.bs = bs
        self.init_spmm(bs, 
                       in_features, out_features,
                       out_features, input_rows, 
                       in_features, input_rows,
                       )
    # spmm API
    def init_spmm(self, batch_num, num_A_rows, num_A_cols,
                                num_B_rows, num_B_cols,
                                num_C_rows, num_C_cols):
        spmm.initSpmmDescriptor(self.index, batch_num,
                                num_A_rows, num_A_cols, num_A_cols, 
                                num_B_rows, num_B_cols, num_B_cols,
                                num_C_rows, num_C_cols, num_C_cols)
    
    # spmm API(check the mask after every update of mask)
    def check_mask(self):
        return spmm.checkPrunned(self.index, self.weight_t_mask) == 0


    def set_origin(self, bool_var):
        self.origin = bool_var

    def set_bpmask(self, mask):
        self.mask = mask
        assert mask.size() == self.weight.size(), "mask size:{}  should be equal to weight size:{}".format(mask.size(), self.weight.size())
        self.weight_t_mask = self.weight.detach().clone() * self.mask
        self.weight_t_mask = self.weight_t_mask.t().cuda().contiguous()

        self.mask.requires_grad = False
        self.check_mask()

        spmm.compressMatrix(self.index, self.weight_t_mask) # need to be check

    def forward(self, x):
        if self.origin:
            return super().forward(x)
        else:
            self.weight_t_mask = self.weight.detach().clone() * self.mask
            self.weight_t_mask = self.weight_t_mask.t().cuda().contiguous()
            output = Matmul_NMBP.apply(self.index, self.weight, x, self.bias, self.weight_t_mask)
            return output


def set_linear_model_origin(model, bool):
    for m in model.modules():
        if isinstance(m, NMBP_Linear):
            m.set_origin(bool)


def update_linear_inter_bpmask(model, train_data, num_samples, param_groups, pattern, n, m, sparsity):
    assert pattern in ['nm']
    fisher_dict = get_fisher(model, train_data, num_samples, param_groups)

    
    module_list = []
    for module in model.modules():
        if isinstance(module, NMBP_Linear):
            module_list.append(module)
    if len(module_list) == 0: raise ValueError
    for module in module_list:
        fisher = fisher_dict[id(module.weight)]
        assert fisher.shape[-1] % 4 == 0, "cusparseLt needs a fixed memory length"
        mask = get_nm_mask(fisher.transpose(-1, -2), n, m)
        module.set_bpmask(mask.transpose(-1, -2))