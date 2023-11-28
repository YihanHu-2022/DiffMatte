import torch.nn as nn
from functools import partial
from detectron2.config import LazyCall as L
from modeling import ViTMatte, DiffusionMattingCriterion, ViT
from modeling.decoder.DUNet import DUNet
from modeling.meta_arch.difmodel import DifModel
from modeling import create_backbone

# Base
# embed_dim, num_heads = 384, 6


model = L(DifModel)(
    backbone = L(create_backbone)(
        name = "ViT",
        kwargs={
            "in_chans": 4,
            "img_size":512,
            "patch_size":16,
            "embed_dim":384,
            "depth":12,
            "num_heads":6,
            "drop_path_rate":0,
            "window_size":14,
            "mlp_ratio":4,
            "qkv_bias":True,
            "norm_layer":partial(nn.LayerNorm, eps=1e-6),
            "window_block_indexes":[
                # 2, 5, 8 11 for global attention
                0,
                1,
                3,
                4,
                6,
                7,
                9,
                10,
            ],
            "residual_block_indexes":[2, 5, 8, 11],
            "use_rel_pos":True,
            "out_feature":"last_feat",
        }
    ),
    # backbone = L(ViT)(  # Single-scale ViT backbone
    #     in_chans=4,
    #     img_size=512,
    #     patch_size=16,
    #     embed_dim=embed_dim,
    #     depth=12,
    #     num_heads=num_heads,
    #     drop_path_rate=0,
    #     window_size=14,
    #     mlp_ratio=4,
    #     qkv_bias=True,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
    #     window_block_indexes=[
    #         # 2, 5, 8 11 for global attention
    #         0,
    #         1,
    #         3,
    #         4,
    #         6,
    #         7,
    #         9,
    #         10,
    #     ],
    #     residual_block_indexes=[2, 5, 8, 11],
    #     use_rel_pos=True,
    #     out_feature="last_feat",
    # ),
    pixel_mean = [123.675 / 255., 116.280 / 255., 103.530 / 255.],
    pixel_std = [58.395 / 255., 57.120 / 255., 57.375 / 255.],
    decoder=L(DUNet)(
        model_channels=32,
        emb_channels=32,
        downsample_in = [7, 32, 64, 128],
        upsample_in = [384, 256, 128, 64, 32],
    ),
    backbone_in_channel = 4,
    decoder_in_channel = 7
)



