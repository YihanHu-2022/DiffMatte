from detectron2.config import LazyCall as L
from modeling.meta_arch.difmatte import DifMatte


use_ddim = True
batch_size = 8
image_size = 512
clip_denoised = True

model = None
diffusion = None

difmatte = L(DifMatte)(
    model = model,
    diffusion = diffusion,
    input_format = "RGB",
    size_divisibility=32,
    args = {
        "use_ddim": use_ddim,
        "batch_size" : batch_size,
        "image_size" : image_size,
        "clip_denoised" : clip_denoised
    }
)