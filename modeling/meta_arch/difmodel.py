import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import cv2
import numpy as np
from detectron2.structures import ImageList

class DifModel(nn.Module):
    def __init__(self,
                 *,
                 backbone,
                 pixel_mean,
                 pixel_std,
                 decoder,
                 backbone_in_channel,
                 decoder_in_channel
                 ):
        super(DifModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.backbone_in_channel = backbone_in_channel
        self.decoder_in_channel = decoder_in_channel
        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
    
    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, features=None, intermidate_out=False):
        images, dimages, timesteps, H, W = self.preprocess_inputs(batched_inputs, decoder_in_channel=self.decoder_in_channel)

        if features == None:
            features = self.backbone(images)
            if intermidate_out:
                outputs = {"feature": features}
                return outputs
            
        outputs = self.decoder(features, dimages, timesteps)
        outputs.update({"feature": features})
        outputs["phas"] = outputs["phas"][:,:,:H,:W]
        outputs["noise"]  = outputs["noise"][:,:,:H,:W]

        return outputs



    def trimap_transform(self, trimap, channels=12, L = 320): # 接收trimap: 0, 0.5, 1
        assert channels in [5, 7, 12]
        if channels == 5:
            return trimap

        trimap = trimap * 2
        trimap = trimap.long()
        trimap = F.one_hot(trimap, num_classes=3).squeeze(1).permute(0, 3, 1, 2).contiguous()
        if channels == 7:
            return trimap

        def dt(a):
            return cv2.distanceTransform((a * 255).astype(np.uint8), cv2.DIST_L2, 0)
        batch = []
        for tri in trimap:
            clicks = []
            tri = tri.permute(1, 2, 0).cpu().numpy()
            for k in [0, 2]:
                dt_mask = -dt(1 - tri[:, :, k])**2
                clicks.append(np.exp(dt_mask / (2 * ((0.02 * L)**2))))
                clicks.append(np.exp(dt_mask / (2 * ((0.08 * L)**2))))
                clicks.append(np.exp(dt_mask / (2 * ((0.16 * L)**2))))
            clicks = np.array(clicks)
            clicks = torch.from_numpy(clicks).float().to(trimap.device)
            batch.append(clicks)
        tritr = torch.stack(batch)
        tri = torch.concat([trimap[:, 0:1], trimap[:, 2:3], tritr], dim=1)
        return tri

    def preprocess_inputs(self, batched_inputs, decoder_in_channel=7, L=320):
        """
        Normalize, pad and batch the input images.
        """
        images = batched_inputs["image"].to(self.device)
        trimap = batched_inputs["trimap"].to(self.device) # train: 0, 0.5, 1
        x_t = batched_inputs["x_t"].to(self.device)
        images = (images - self.pixel_mean) / self.pixel_std

        if decoder_in_channel == 4:
            cond = images
        else:
            tri = self.trimap_transform(trimap, decoder_in_channel, L)
            cond = torch.cat([images, tri], dim = 1)
        cond = cond * 2 - 1   # condition normalize
        dimages = torch.cat([cond, x_t], dim = 1)

        if self.backbone_in_channel == 4:
            images = torch.cat((images, trimap), dim=1) 
        elif self.backbone_in_channel == 6:
            tri = trimap * 2
            tri = tri.long()
            tri = F.one_hot(tri, num_classes=3).squeeze(1).permute(0, 3, 1, 2).contiguous()
            images = torch.cat((images, tri), dim=1) 
        
        B, C, H, W = images.shape
        if images.shape[-1]%32!=0 or images.shape[-2]%32!=0:
            new_H = (32-images.shape[-2]%32) + H
            new_W = (32-images.shape[-1]%32) + W
            new_images = torch.zeros((images.shape[0], images.shape[1], new_H, new_W)).to(self.device)
            new_images[:,:,:H,:W] = images[:,:,:,:]

            dnew_images = torch.zeros((dimages.shape[0], dimages.shape[1], new_H, new_W)).to(self.device)
            dnew_images[:,:,:H,:W] = dimages[:,:,:,:]
            
            del images
            del dimages

            images = new_images
            dimages = dnew_images

        return images, dimages, batched_inputs["timestep"], H, W