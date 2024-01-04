from monai.data import Dataset, CacheDataset, SmartCacheDataset

SYNAPSE_METAINFO = dict(
    classes=('background', 'spleen', 'right_kidney', 'left_kidney',
             'gallbladder', 'esophagus', 'liver', 'stomach',
             'aorta', 'inferior_vena_cava', 'portal_vein_and_splenic_vein', 'pancreas',
             'right_adrenal_gland', 'left_adrenal_gland'),
    palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
             [255, 255, 0], [0, 255, 255], [255, 0, 255], [255, 239, 213],
             [0, 0, 205], [205, 133, 63], [210, 180, 140], [102, 205, 170],
             [0, 0, 128], [0, 139, 139]])

KITS23_METAINFO = dict(
    classes=('background', 'kidney', 'tumor', 'cyst'),
    palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])

BRATS21_METAINFO = dict(
    classes=('TC', 'WT', 'ET'),
    palette=[[255, 0, 0], [0, 255, 0], [0, 0, 255]])

BRATS23_METAINFO = dict(
    classes=('background', 'TC', 'WT', 'ET'),
    palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])

WORD_METAINFO = dict(
    classes=('background', 'liver', 'spleen', 'left_kidney',
             'right_kidney', 'stomach', 'gallbladder', 'esophagus',
             'pancreas', 'duodenum', 'colon', 'intestine',
             'adrenal', 'rectum', 'bladder', 'Head_of_femur_L',
             'Head_of_femur_R'),
    palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
             [255, 255, 0], [0, 255, 255], [255, 0, 255], [255, 239, 213],
             [0, 0, 205], [205, 133, 63], [210, 180, 140], [102, 205, 170],
             [0, 0, 128], [0, 139, 0], [139, 0, 139], [139, 139, 0],
             [139, 0, 0]])

class MonaiDataset(Dataset):

    def __init__(self, meta_info, **kwargs):
        super().__init__(**kwargs)
        self.metainfo = meta_info


class CacheMonaiDataset(CacheDataset):

    def __init__(self, meta_info, **kwargs):
        super().__init__(**kwargs)
        self.metainfo = meta_info

    def clear_cache(self):
        del self._cache


class SmartCacheMonaiDataset(SmartCacheDataset):

    def __init__(self, meta_info, **kwargs):
        super().__init__(**kwargs)
        self.metainfo = meta_info

    def clear_cache(self):
        del self._cache
