# Copyright (c) OpenMMLab. All rights reserved.
"""MMSegmentation provides 17 registry nodes to support using modules across
projects. Each node is a child of the root registry in MMEngine.

More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""
from mmseg.registry import DATA_SAMPLERS as MMSEG_DATA_SAMPLERS
from mmseg.registry import DATASETS as MMSEG_DATASETS
from mmseg.registry import EVALUATOR as MMSEG_EVALUATOR
from mmseg.registry import HOOKS as MMSEG_HOOKS
from mmseg.registry import INFERENCERS as MMSEG_INFERENCERS
from mmseg.registry import LOG_PROCESSORS as MMSEG_LOG_PROCESSORS
from mmseg.registry import LOOPS as MMSEG_LOOPS
from mmseg.registry import METRICS as MMSEG_METRICS
from mmseg.registry import MODEL_WRAPPERS as MMSEG_MODEL_WRAPPERS
from mmseg.registry import MODELS as MMSEG_MODELS
from mmseg.registry import \
    OPTIM_WRAPPER_CONSTRUCTORS as MMSEG_OPTIM_WRAPPER_CONSTRUCTORS
from mmseg.registry import OPTIM_WRAPPERS as MMSEG_OPTIM_WRAPPERS
from mmseg.registry import OPTIMIZERS as MMSEG_OPTIMIZERS
from mmseg.registry import PARAM_SCHEDULERS as MMSEG_PARAM_SCHEDULERS
from mmseg.registry import \
    RUNNER_CONSTRUCTORS as MMSEG_RUNNER_CONSTRUCTORS
from mmseg.registry import RUNNERS as MMSEG_RUNNERS
from mmseg.registry import TASK_UTILS as MMSEG_TASK_UTILS
from mmseg.registry import TRANSFORMS as MMSEG_TRANSFORMS
from mmseg.registry import VISBACKENDS as MMSEG_VISBACKENDS
from mmseg.registry import VISUALIZERS as MMSEG_VISUALIZERS
from mmseg.registry import \
    WEIGHT_INITIALIZERS as MMSEG_WEIGHT_INITIALIZERS

from mmengine.registry import Registry

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry('runner', parent=MMSEG_RUNNERS)
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor', parent=MMSEG_RUNNER_CONSTRUCTORS)
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry('loop', parent=MMSEG_LOOPS)
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry(
    'hook', parent=MMSEG_HOOKS, locations=['seg.engine.hooks'])

# manage data-related modules
DATASETS = Registry(
    'dataset', parent=MMSEG_DATASETS, locations=['seg.datasets'])
DATA_SAMPLERS = Registry(
    'data sampler',
    parent=MMSEG_DATA_SAMPLERS)
TRANSFORMS = Registry(
    'transform',
    parent=MMSEG_TRANSFORMS)

# mangage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', parent=MMSEG_MODELS, locations=['seg.models'])
# mangage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=MMSEG_MODEL_WRAPPERS,
    locations=['seg.models'])
# mangage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=MMSEG_WEIGHT_INITIALIZERS,
    locations=['seg.models'])

# mangage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry(
    'optimizer',
    parent=MMSEG_OPTIMIZERS)
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry(
    'optim_wrapper',
    parent=MMSEG_OPTIM_WRAPPERS)
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer wrapper constructor',
    parent=MMSEG_OPTIM_WRAPPER_CONSTRUCTORS)
# mangage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=MMSEG_PARAM_SCHEDULERS)

# manage all kinds of metrics
METRICS = Registry(
    'metric', parent=MMSEG_METRICS, locations=['seg.evaluation'])
# manage evaluator
EVALUATOR = Registry(
    'evaluator', parent=MMSEG_EVALUATOR, locations=['seg.evaluation'])

# manage task-specific modules like ohem pixel sampler
TASK_UTILS = Registry(
    'task util', parent=MMSEG_TASK_UTILS, locations=['seg.models'])

# manage visualizer
VISUALIZERS = Registry(
    'visualizer',
    parent=MMSEG_VISUALIZERS,
    locations=['seg.visualization'])
# manage visualizer backend
VISBACKENDS = Registry(
    'vis_backend',
    parent=MMSEG_VISBACKENDS,
    locations=['seg.visualization'])

# manage logprocessor
LOG_PROCESSORS = Registry(
    'log_processor',
    parent=MMSEG_LOG_PROCESSORS,
    locations=['seg.visualization'])

# manage inferencer
INFERENCERS = Registry('inferencer', parent=MMSEG_INFERENCERS)
