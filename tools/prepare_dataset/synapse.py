# case.h5.py to slice.npz
import os
import os.path as osp
import numpy as np
import h5py
test_dir = '/root/ex_kd/data/synapse9/test_vol_h5'
test_npz = '/root/ex_kd/data/synapse9/test_npz'
for case in os.listdir(test_dir):
    data = h5py.File(osp.join(test_dir, case))
    img, label = data['image'][:], data['label'][:]
    slice_nums = img.shape[0]
    for i in range(img.shape[0]):
        img_i = img[i]
        label_i = label[i]
        np.savez(
            osp.join(test_npz, case.split('.')[0] + '_slice' + str(i).zfill(3) + '.npz'),
            image=img_i,
            label=label_i)
