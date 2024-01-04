from seg.evaluation.metrics.monai_metric import MonaiMetric
from seg.evaluation.monai_evaluator import MonaiEvaluator
from seg.datasets.monai_dataset import WORD_METAINFO
roi = [96, 96, 96]

dataloader_cfg = dict(
    data_name='WORD',
    data_dir='data/WORD',
    json_list='dataset.json',
    train_case_nums=100,
    meta_info=WORD_METAINFO,
    # use monai Dataset class
    use_normal_dataset=False,
    use_smart_dataset=False,
    # batch size and worker
    batch_size=1,
    num_samples=4,
    workers=8,
    distributed=False,
    # spacing
    space_x=1.5,
    space_y=1.5,
    space_z=2.0,
    # ScaleIntensityRanged
    a_min=-175.0,
    a_max=250,
    b_min=0.0,
    b_max=1.0,
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
        # metrics=['Dice', 'IoU'],
        num_classes=17))
test_evaluator = val_evaluator