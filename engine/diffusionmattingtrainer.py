from detectron2.engine import AMPTrainer
import torch
import time
import logging
import detectron2.utils.comm as comm

from modeling.diffusion.resample import LossAwareSampler
from detectron2.utils.events import EventWriter, get_event_storage

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class DiffusionMattingTrainer(AMPTrainer):
    def __init__(
        self, 
        model, 
        data_loader, 
        optimizer, 
        schedule_sampler=None,
        log_image_period=1000,
        self_align_stage1_step = 1e8,
        self_align_stage2_step = 1e8,
    ):
        super().__init__(model, data_loader, optimizer, grad_scaler=None)
        self.data_loader_iter = iter(cycle(self.data_loader))
        self.schedule_sampler = schedule_sampler
        self.log_image_period = log_image_period
        self.self_align_stage1_step = self_align_stage1_step
        self.self_align_stage2_step = self_align_stage2_step
    
    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        #matting pass
        start = time.perf_counter()        
        data = next(self.data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast():
            if self.schedule_sampler != None:
                t, weights = self.schedule_sampler.sample(data["image"].shape[0], data["image"].device)
                loss_dict, output_images = self.model(data, t)
            else:
                if self.iter < self.self_align_stage1_step and self.iter < self.self_align_stage2_step:
                    self_align_stage1 = False # temp_times = 1的self_align阶段
                    self_align_stage2 = False # temp_times = [0,1]的self_align阶段
                elif self.iter >= self.self_align_stage1_step and self.iter < self.self_align_stage2_step:
                    self_align_stage1 = True
                    self_align_stage2 = False
                else:  
                    self_align_stage1 = False
                    self_align_stage2 = True
                loss_dict, output_images, t = self.model(data, self_align_stage1 = self_align_stage1, self_align_stage2 = self_align_stage2)

            total_loss = 0.
            for key in loss_dict.keys():
                total_loss = total_loss + loss_dict[key]
            
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, total_loss.detach()
                )
                weights = weights.to(total_loss.device)
                loss = (total_loss * weights).mean()
            else:
                loss = total_loss.mean()

        
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()

        self._write_metrics(loss_dict, data_time)
        self._write_images(t, output_images, data)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
    
    def _write_images(self, t, output_images: torch.Tensor, data: torch.Tensor, iter: int = None):
        logger = logging.getLogger(__name__)
        iter = self.iter if iter is None else iter
        storage = get_event_storage()
        storage._t = ()
        if (iter + 1) % self.log_image_period == 0:
            try:
                DiffusionMattingTrainer.write_images(t, output_images, data, iter)
            except Exception:
                logger.exception("Exception in writing images: ")
                raise
    
    @staticmethod
    def write_images(t, output_images: torch.Tensor, data: torch.Tensor, cur_iter:int = None):
        # output_images = output_images.detach().cpu()
        if comm.is_main_process():
            storage = get_event_storage()
            storage.put_image("fg", data["fg"])
            storage.put_image("alpha", data["alpha"])
            storage.put_image("bg", data["bg"])
            storage._t = (str(t), storage.iter)
            for key in output_images.keys():
                storage.put_image(key, output_images[key])