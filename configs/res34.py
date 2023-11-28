
from .common.train import train
from .common.model import model
from .common.diffusion import diffusion
from .common.difmatte import difmatte
from .common.optimizer import optimizer
from .common.scheduler import lr_multiplier
from .common.dataloader import dataloader
from .common.dataloader import train_dataset
from .common.loss import loss
from modeling.backbone.res34 import BasicBlock

train.max_iter = int(43100 / 20 / 2 * 150)
train.checkpointer.period = int(43100 / 20 / 2 * 10)
train.self_align_stage1_step = int(43100 / 20 / 2 * 1000)
train.self_align_stage2_step = int(43100 / 20 / 2 * 100)

optimizer.lr=4e-4
lr_multiplier.scheduler.values=[1.0, 0.1, 0.05, 0.01]
lr_multiplier.scheduler.milestones=[int(43100 / 20 / 2 * 30), int(43100 / 20 / 2 * 90), int(43100 / 20 / 2 * 140)]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter

train.init_checkpoint = 'INIT_CHECKPOINT'
train.output_dir = './output_of_train/res34'


dataloader.train.batch_size = 20

model.backbone.name = "Res34"
model.backbone.kwargs = {
    "block": BasicBlock,
    "in_channels" : 6,
    "layers": [3, 4, 4, 2],
}
model.backbone_in_channel = 6
model.decoder_in_channel = 7

model.decoder.downsample_in = [7, 32, 64, 128, 128]
model.decoder.upsample_in = [512, 256, 256, 128, 64, 32]

dataloader.train.num_workers = 4

loss.use_mse = True
loss.use_mat = True

diffusion.steps = 1
diffusion.noise_schedule = "linear"
diffusion.scale = 0.2

difmatte.args["use_ddim"] = False