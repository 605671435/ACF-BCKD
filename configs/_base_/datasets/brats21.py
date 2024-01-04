from seg.evaluation.metrics.monai_metric import MonaiMetric
from seg.evaluation.monai_evaluator import MonaiEvaluator
from seg.datasets.monai_dataset import BRATS21_METAINFO
roi = [96, 96, 96]

dataloader_cfg = dict(
    data_name='BraTS21',
    data_dir='data/brats21',
    json_list='dataset_0.json',
    train_case_nums=1001,
    meta_info=BRATS21_METAINFO,
    # use monai Dataset class
    use_normal_dataset=True,
    # batch size and worker
    batch_size=4,
    workers=8,
    distributed=False,
    # spacing
    space_x=1.5,
    space_y=1.5,
    space_z=2.0,
    # roi size
    roi_x=roi[0],
    roi_y=roi[1],
    roi_z=roi[2],
    # RandFlipd aug probability
    RandFlipd_prob=0.2,
    # RandRotate90d aug probability
    RandRotate90d_prob=0.2,
    # RandScaleIntensityd aug probability
    RandScaleIntensityd_prob=0.1,
    # RandShiftIntensityd aug probability
    RandShiftIntensityd_prob=0.1
)

val_evaluator = dict(
    type=MonaiEvaluator,
    metrics=dict(
        type=MonaiMetric,
        metrics=['Dice', 'HD95'],
        num_classes=3,
        one_hot=False,
        include_background=True,
        reduction='mean_batch',
        print_per_class=False))
test_evaluator = val_evaluator
