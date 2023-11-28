'''
Inference for Composition-1k Dataset.
'''
import os
import cv2
from re import findall
import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from os.path import join as opj
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser

import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


#Dataset and Dataloader
def collate_fn(batched_inputs):
    rets = dict()
    for k in batched_inputs[0].keys():
        rets[k] = torch.stack([_[k] for _ in batched_inputs])
    return rets

class Composition_1k(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_names = sorted(os.listdir(opj(self.data_dir, 'merged')))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        imgs = cv2.imread(opj(self.data_dir, 'merged', self.file_names[idx]))[:,:,::-1]
        imgs = imgs.transpose((2, 0, 1)).astype(np.float32) / 255.

        if '.jpg' in self.file_names[idx]:
            self.file_names[idx] = self.file_names[idx][:-4] + '.png'
        # print(opj(self.data_dir, 'merged', self.file_names[idx]))

        tris = cv2.imread(opj(self.data_dir, 'trimaps', self.file_names[idx]), 0)
        tris[tris < 85] = 0
        tris[tris >= 170] = 2
        tris[tris >= 85] = 1

        sample = {}

        sample['trimap'] = torch.from_numpy(tris)[None, ...].to(torch.long).float() / 2
        sample['image'] = torch.from_numpy(imgs)
        sample['image_name'] = self.file_names[idx]

        return sample

#model and output
def matting_inference(
    config_dir='',
    checkpoint_dir='',
    inference_dir='',
    data_dir='',
    rank = None,
    sample_strategy = None,
):
    #initializing model
    cfg = LazyConfig.load(config_dir)
    seed_everything(cfg.train.seed)
    if sample_strategy is not None:
        cfg.difmatte.args["use_ddim"] = True if "ddim" in sample_strategy else False
        cfg.diffusion.steps = int(findall(r"\d+", sample_strategy)[0])
    model = instantiate(cfg.model)
    diffusion = instantiate(cfg.diffusion)
    cfg.difmatte.model = model
    cfg.difmatte.diffusion = diffusion
    difmatte = instantiate(cfg.difmatte)
    difmatte.to(cfg.train.device if rank is None else rank)
    difmatte.eval()
    DetectionCheckpointer(difmatte).load(checkpoint_dir)

    #initializing dataset
    composition_1k_dataloader = DataLoader(
    dataset = Composition_1k(
        data_dir = data_dir
    ),
    shuffle = False,
    batch_size = 1,
    # collate_fn = collate_fn,
    )
    
    #inferencing
    os.makedirs(inference_dir, exist_ok=True)

    for data in tqdm(composition_1k_dataloader):
        with torch.no_grad():
            data["trimap"] = data["trimap"].to(difmatte.model.device if rank is None else rank)
            data["image"] = data["image"].to(difmatte.model.device if rank is None else rank)
            output = difmatte(data)
            _al = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(opj(inference_dir, data['image_name'][0]), _al)
            torch.cuda.empty_cache()

if __name__ == '__main__':
    #add argument we need:
    parser = default_argument_parser()
    parser.add_argument('--config-dir', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--inference-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--sample-strategy', type=str, default="ddim10")
    
    args = parser.parse_args()
    matting_inference(
        config_dir = args.config_dir,
        checkpoint_dir = args.checkpoint_dir,
        inference_dir = args.inference_dir,
        data_dir = args.data_dir,
        sample_strategy = args.sample_strategy
    )