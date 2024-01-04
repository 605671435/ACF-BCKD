from seg.evaluation.metrics.monai_metric import MonaiMetric
from seg.evaluation.monai_evaluator import MonaiEvaluator
from seg.datasets.monai_dataset import SYNAPSE_METAINFO
roi = [128, 128, 96]
spacing = [
    2.0,
    1.5,
    1.5]
# spacing = [
#     1.0,
#     1.0,
#     1.0]
dataloader_cfg = dict(
    data_dir='data/synapse_raw',
    json_list='dataset_0.json',
    train_case_nums=24,
    meta_info=SYNAPSE_METAINFO,
    # use monai Dataset class
    use_normal_dataset=False,
    # batch size and worker
    batch_size=1,
    num_samples=4,
    workers=8,
    distributed=False,
    # spacing
    space_x=spacing[2],
    space_y=spacing[1],
    space_z=spacing[0],
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
        metrics=['Dice', 'IoU', 'HD95'],
        # metrics=['Dice', 'IoU'],
        num_classes=14))
test_evaluator = val_evaluator