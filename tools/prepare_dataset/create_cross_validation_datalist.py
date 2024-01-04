import os
import os.path as osp

from monai.data import create_cross_validation_datalist


def main():
    img_list = os.listdir('data/Dataset220_KiTS2023/imagesTr')
    label_list = os.listdir('data/Dataset220_KiTS2023/labelsTr')
    img_list.sort()
    label_list.sort()

    datalist = [{'image': 'imagesTr/' + img, 'label': 'labelsTr/' + lb} for img, lb in zip(img_list, label_list)]
    create_cross_validation_datalist(
        datalist=datalist,
        nfolds=5,
        train_folds=[0, 1, 2, 3],
        val_folds=[4],
        shuffle=False,
        filename='data/Dataset220_KiTS2023/dataset_0.json')

def brats21():
    base_dir = 'data/TrainingData'
    img_list = os.listdir(base_dir)

    img_list.sort()
    datalist = []
    for img in img_list:
        image = [
            osp.join('TrainingData', img, img + '-t1c.nii.gz'),
            osp.join('TrainingData', img, img + '-t1n.nii.gz'),
            osp.join('TrainingData', img, img + '-t2f.nii.gz'),
            osp.join('TrainingData', img, img + '-t2w.nii.gz'),
        ]
        for nii in image:
            assert osp.exists(osp.join('data', nii))
        label = osp.join('TrainingData', img, img + '-seg.nii.gz')
        assert osp.exists(osp.join('data', label))
        datalist.append({'image': image, 'label': label})

    create_cross_validation_datalist(
        datalist=datalist,
        nfolds=5,
        train_folds=[0, 1, 2, 3],
        val_folds=[4],
        shuffle=False,
        filename='data/dataset_0.json')

if __name__ == '__main__':
    brats21()
