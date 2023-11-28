from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from detectron2.config import LazyCall as L
from torch.utils.data.distributed import DistributedSampler

from data import ImageFileTrain, DataGenerator, ImageFileTest

#Dataloader
train_dataset = L(DataGenerator)(
    data = L(ImageFileTrain)(
        alpha_dir='DataDir/AdobeImageMatting/train/alpha',
        fg_dir='DataDir/AdobeImageMatting/train/fg',
        bg_dir='DataDir/Segmentation/COCO/train2014',
        root='DataDir'
    ),
    phase = 'train',
    crop_size = 1024
)

dataloader = OmegaConf.create()
dataloader.train = L(DataLoader)(
    dataset = train_dataset,
    batch_size=15,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    sampler=L(DistributedSampler)(
        dataset = train_dataset,
    ),
    drop_last=True
)