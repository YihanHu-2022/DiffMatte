from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler, CosineParamScheduler

def create_ParamScheduler(name, **kwargs):
    ParamScheduler = {
        "MultiStep": MultiStepParamScheduler,
        "Cosine": CosineParamScheduler
    }[name]
    return ParamScheduler(**kwargs)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(create_ParamScheduler)(
        name="MultiStep",
        # values = [1.0, 0.1, 0.01],
        # milestones = [96778, 103579],
        # num_updates = 100
    ),
    warmup_length=250 / 100,
    warmup_factor=0.001,
)