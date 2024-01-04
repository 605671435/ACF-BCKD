# modified from https://github.com/MenghaoGuo/EANet/blob/main/EAMLP/models/token_transformer.py

# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.

import torch.nn as nn

class External_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.dim = dim
        self.k = 128

        self.q_linear = nn.Linear(self.dim, self.in_dim)

        self.linear_1 = nn.Linear(self.in_dim, self.k, bias=False)

        self.linear_2 = nn.Linear(self.k, self.in_dim)
        self.linear_2.weight.data = self.linear_1.weight.data.permute(1, 0)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.in_dim, self.in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, h * w, c)

        x = self.q_linear(x)
        idn = x[:]

        x = self.linear_1(x)
        x = x.softmax(dim=1)
        x = x / (1e-9 + x.sum(dim=-1, keepdim=True))  #
        x = self.attn_drop(x)
        x = self.linear_2(x)

        x = self.proj(x)  # add offset
        x = self.proj_drop(x)

        # skip connection
        x = idn + x  # because the original x has different size with current x, use v to do skip connection
        return x.view(b, c, h, w)
