from seg.evaluation.metrics.monai_metric import MonaiMetric
from seg.evaluation.monai_evaluator import MonaiEvaluator
from seg.datasets.monai_dataset import KITS23_METAINFO
roi = [96, 96, 96]

dataloader_cfg = dict(
    data_name='KiTS2023',
    data_dir='data/Dataset220_KiTS2023',
    json_list='dataset_0.json',
    # train_case_nums=392,
    train_case_nums=100,
    meta_info=KITS23_METAINFO,
    # use monai Dataset class
    use_normal_dataset=False,
    # batch size and worker
    batch_size=1,
    num_samples=4,
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
)

val_evaluator = dict(
    type=MonaiEvaluator,
    metrics=dict(
        type=MonaiMetric,
        metrics=['Dice', 'HD95'],
        num_classes=4))
test_evaluator = val_evaluator