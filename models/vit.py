# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from utils.configurable import configurable
from models.build import MODELS_REGISTRY
from models.nm_linear import NMBP_Linear

import spmm
from solver.utils import get_nm_mask

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, 
        image_size, patch_size, num_classes, 
        dim, depth, heads, mlp_dim, 
        pool = 'cls', 
        channels = 3, dim_head = None, dropout = 0., emb_dropout = 0.
    ):
        super().__init__()
        self.depth = depth
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        dim_head = dim // heads
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + (pool == 'cls'), dim))
        self.n_patch = self.pos_embedding.shape[1]

        if pool == 'cls': 
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        if self.pool == 'cls' :
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


def _cfg_to_vit(args):
    return {
        "num_classes": args.n_classes,
        "patch_size": args.patch_size,
        "culinear": args.culinear,
        "batch_size": args.batch_size,
    }



@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_vit)
def vit_testspmm(
    patch_size, num_classes, culinear, batch_size
):
    model = ViT(
        image_size = 32,
        patch_size = patch_size,
        num_classes = num_classes,
        dim = 8192,
        depth = 1,
        heads = 1,
        mlp_dim = 8192,
        dropout = 0,
        emb_dropout = 0,
        pool = 'mean',
    )
    

    if culinear:
        spmm.initSpmmNum(4 * model.depth + 2 + 1) # must larger than 4 \times depth + 2
        if spmm.checkCusparseLt() != 0:
            raise RuntimeError("Hardware does not support cusparseLt repo.") 
        
        assert model.n_patch % 4 == 0, "To use cusparseLt, num_patches should be divisible by four "
        index = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if name == 'to_patch_embedding.1' or name == 'mlp_head.1': 
                    continue
                module.__class__ = NMBP_Linear
                in_f, out_f, bias = module.in_features, module.out_features, module.bias
                print('Using spmm speed up, transforming the linear layer whose name is {} and input_dim is {}, output_dim is {}, n_patch is {}.'.format(name, in_f, out_f, model.n_patch))
                module.__init__(
                    index, batch_size, 
                    in_f, out_f, model.n_patch,
                    bias=True if bias is not None else False)
                index += 1

                # init the spmm mask
                mask = get_nm_mask(torch.rand(in_f, out_f), 1, 2)
                module.set_bpmask(mask.t())
        print('transform done.')
    return model

# run script for vit on cifar:
# python train.py --model vit_tiny --dataset CIFAR10_cutout --datadir [Path2Data] --opt adam --lr 1e-4 --weight_decay 0 --seed 1234 --batch_size 512 --epochs 200

# script for vit with cusparselt:
# CUDA_VISIBLE_DEVICES=1 python train.py \
# --model vit_nodo --dataset CIFAR100_base --datadir .. \
# --opt sam-adam --rho 0.2 --lr 1e-4 --weight_decay 0 --seed 1234 --batch_size 512 --epochs 200 --output_dir logs \
# --n_structured 1 --m_structured 2 --structured nm --implicit --culinear