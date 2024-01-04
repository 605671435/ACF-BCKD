# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from numpy.random import randint


def patch_rand_drop(x, x_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    c, h, w = x.size()
    n_drop_pix = np.random.uniform(0, max_drop) * h * w
    mx_blk_height = int(h * max_block_sz)
    mx_blk_width = int(w * max_block_sz)
    tolr = (int(tolr * h), int(tolr * w))
    total_pix = 0
    while total_pix < n_drop_pix:
        rnd_r = randint(0, h - tolr[0])
        rnd_c = randint(0, w - tolr[1])
        rnd_h = min(randint(tolr[0], mx_blk_height) + rnd_r, h)
        rnd_w = min(randint(tolr[1], mx_blk_width) + rnd_c, w)
        if x_rep is None:
            x_uninitialized = torch.empty(
                (c, rnd_h - rnd_r, rnd_w - rnd_c), dtype=x.dtype, device=x.device
            ).normal_()
            x_uninitialized = (x_uninitialized - torch.min(x_uninitialized)) / (
                torch.max(x_uninitialized) - torch.min(x_uninitialized)
            )
            x[:, rnd_r:rnd_h, rnd_c:rnd_w] = x_uninitialized
        else:
            x[:, rnd_r:rnd_h, rnd_c:rnd_w] = x_rep[:, rnd_r:rnd_h, rnd_c:rnd_w]
        total_pix = total_pix + (rnd_h - rnd_r) * (rnd_w - rnd_c)
    return x


def rot_rand(x_s):
    img_n = x_s.size()[0]
    x_aug = x_s.detach().clone()
    x_rot = torch.zeros(img_n).long().to(x_s)
    for i in range(img_n):
        x = x_s[i]
        orientation = np.random.randint(0, 3)
        if orientation == 0:
            pass
        elif orientation == 1:
            x = x.rot90(1, (1, 2))
        elif orientation == 2:
            x = x.rot90(2, (1, 2))
        x_aug[i] = x
        x_rot[i] = orientation
    return x_aug, x_rot


def aug_rand(samples):
    img_n = samples.size()[0]
    x_aug = samples.detach().clone()
    for i in range(img_n):
        x_aug[i] = patch_rand_drop(x_aug[i])
        idx_rnd = randint(0, img_n)
        if idx_rnd != i:
            x_aug[i] = patch_rand_drop(x_aug[i], x_aug[idx_rnd])
    return x_aug
