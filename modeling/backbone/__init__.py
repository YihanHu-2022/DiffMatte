from .backbone import *
from .vit import ViT
from .vit_grid import ViT as ViT_grid
from .res34 import *
from .swin import *
from .utils import *

def create_backbone(name, kwargs):
    backbone = {
        "ViT": ViT,
        "ViT_d646": ViT_grid,
        "Res34": ResNet_D,
        "Swin": Swin
    }[name]
    return backbone(**kwargs)