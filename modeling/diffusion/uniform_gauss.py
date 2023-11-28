import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import expm1
from functools import partial
import math
from PIL import Image
import numpy as np
import os
from einops import rearrange, reduce, repeat
import enum

from tqdm.auto import tqdm
import cv2


class ModelMeanType(enum.Enum):
    """
    Which type of output torche model predicts.
    """

    PREVIOUS_X = enum.auto()  # torche model predicts x_{t-1}
    START_X = enum.auto()  # torche model predicts x_0
    EPSILON = enum.auto()  # torche model predicts epsilon


def get_unknown_tensor(trimap):
    """
    get 1-channel unknown area tensor from the 3-channel/1-channel trimap tensor
    """
    if trimap.shape[1] == 3:
        weight = trimap[:, 1:2, :, :].float()
    else:
        weight = trimap.eq(1).float()
    return weight

# normalize and unnormalize image

def normalize_img(x):
    return x * 2 - 1

def unnormalize_img(x):
    return (x + 1) * 0.5

def identity(x):
    return x



def safe_div(numer, denom, eps = 1e-10):
    return numer / denom.clamp(min = eps)

# normalize variance of noised image, if scale is not 1

def normalize_img_variance(x, eps = 1e-5):
    std = reduce(x, 'b c h w -> b 1 1 1', partial(torch.std, unbiased = False))
    return x / std.clamp(min = eps)

# helper functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# noise schedules

def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = torch.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)

def power_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = start ** power
    v_end = end ** power
    output = (t * (end - start) + start) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

# converting gamma to alpha, sigma or logsnr

def gamma_to_alpha_sigma(gamma, scale = 1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)

def gamma_to_log_snr(gamma, scale = 1, eps = 1e-5):
    return log(gamma * (scale ** 2) / (1 - gamma), eps = eps)


class UniformGauss():
    def __init__(
        self,
        model_mean_type,
        criterion,
        *,
        timesteps = 1000,
        uniform_timesteps = True,
        noise_schedule = None,
        schedule_kwargs: dict = dict(),
        time_difference = 0.,
        scale = 1.,                      # this will be set to < 1. for better convergence when training on higher resolution images
        min_snr_loss_weight = True,
        min_snr_gamma = 5,
        jump_step = -1,
    ):
        super().__init__()
        self.model_mean_type = model_mean_type
        self.criterion = criterion
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma
        self.uniform_timesteps = uniform_timesteps
        # print("=======================================")
        # print(f"noise_schedule:{noise_schedule}")
        # print("=======================================")

        if noise_schedule == "linear":
            self.gamma_schedule = simple_linear_schedule
        elif noise_schedule == "cosine":
            self.gamma_schedule = cosine_schedule
        elif noise_schedule == "sigmoid":
            self.gamma_schedule = sigmoid_schedule
        elif noise_schedule == "power":
            self.gamma_schedule = power_schedule
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        # the main finding presented in Ting Chen's paper - that higher resolution images requires more noise for better training

        assert scale <= 1, 'scale must be less than or equal to 1'
        self.scale = scale
        self.maybe_normalize_img_variance = normalize_img_variance if scale < 1 else identity

        # gamma schedules

        self.gamma_schedule = partial(self.gamma_schedule, **schedule_kwargs)

        self.timesteps = timesteps
        self.jump_step = jump_step
        if self.jump_step >= self.timesteps:
            raise ValueError(f'jump step must lower than timestep !!!')

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400

        self.time_difference = time_difference

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        if self.jump_step > 0:
            times_c = torch.linspace(1., 0., self.timesteps + 1, device = 'cpu')
            times_c = repeat(times_c, 't -> b t', b = batch)
            times = torch.stack((times_c[:, [0, self.jump_step - 1]].to(device), times_c[:, [self.jump_step, self.timesteps-1]].to(device)), dim = 0)
        times = times.unbind(dim = -1)
        return times

    @torch.no_grad()
    def ddpm_sample(
        self, 
        model, 
        shape, 
        condition,
        noise=None,
        device=None,
        sample_list=None,
        GTalpha=None
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        # trimap = condition["trimap"]
        batch = shape[0]

        if noise is not None:
            noised_pha = noise
        else:
            noised_pha = torch.randn(*shape, device=device)

        time_pairs = self.get_sampling_timesteps(batch, device = device)


        condition["features"] = None
        counter = 0
        interm = {}
        for time, time_next in tqdm(time_pairs, desc = 'ddpm sampling loop time step', total = self.timesteps):

            # add the time delay

            time_next = (time_next - self.time_difference).clamp(min = 0.)

            model_input = condition

            maybe_normalized_pha = self.maybe_normalize_img_variance(noised_pha)

            # maybe_normalized_pha[trimap == 0] = 0
            # maybe_normalized_pha[trimap == 1] = 1

            model_input.update({"x_t":maybe_normalized_pha})
            model_input.update({"timestep":time})
            

            # get predicted x0
            if self.model_mean_type == ModelMeanType.START_X:
                # model_output = model(input)["phas"]
                model_output = model(model_input, condition["features"])
                pred, features = model_output["phas"], model_output["feature"]
            elif self.model_mean_type == ModelMeanType.EPSILON:
                # model_output = model(input)["noise"]
                model_output = model(model_input, condition["features"])
                pred, features = model_output["noise"], model_output["feature"]

            condition["features"] = features # 下一个step重复利用features

            # get log(snr)
            gamma = self.gamma_schedule(time)
            gamma_next = self.gamma_schedule(time_next)
            gamma, gamma_next = map(partial(right_pad_dims_to, noised_pha), (gamma, gamma_next))

            # get alpha sigma of time and next time

            alpha, sigma = gamma_to_alpha_sigma(gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(gamma_next, self.scale)

            # calculate x0 and noise

            if self.model_mean_type == ModelMeanType.START_X: #case2 model预测x0
                pred_xstart = pred
            else: #case3 预测eps的期望值
                pred_xstart = safe_div(noised_pha - sigma * pred, alpha)
                # pred_xstart = safe_div(maybe_normalized_pha - sigma * pred, alpha)


            # clip x0
            pred_xstart.clamp_(-1., 1.)
            if sample_list:
                if counter in sample_list:
                    alpha_pred = unnormalize_img(pred_xstart)
                    alpha_pred = torch.clamp(alpha_pred, 0., 1.)
                    trimap = condition["trimap"]
                    alpha_pred[trimap == 0] = 0.0
                    alpha_pred[trimap == 2] = 1.0

                    alpha_pred = alpha_pred[0, 0, ...].data.cpu().numpy() * 255
                    interm.update({"step_"+str(counter): alpha_pred})
                if counter == sample_list[-1]:
                    return interm
            counter += 1

            # derive posterior mean and variance

            log_snr, log_snr_next = map(gamma_to_log_snr, (gamma, gamma_next))

            c = -expm1(log_snr - log_snr_next)

            if GTalpha == None:
                mean = alpha_next * (noised_pha * (1 - c) / alpha + c * pred_xstart)
            else:
                GTpha = normalize_img(GTalpha)
                mean = alpha_next * (noised_pha * (1 - c) / alpha + c * GTpha)

            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # get noise

            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(noised_pha),
                torch.zeros_like(noised_pha)
            ) # no noise when t==0

            noised_pha = mean + (0.5 * log_variance).exp() * noise
            # cv2.imwrite("/home/yihan.hu/learner/DiffusionMattingV3/detail/noised_pha_pre"+str(time.item())+".png", noised_pha.cpu().numpy() * 255)

        if sample_list != None and counter == sample_list[-1]:
            pred_xstart = unnormalize_img(noised_pha / self.scale)
            alpha_pred = torch.clamp(pred_xstart, 0., 1.)
            trimap = condition["trimap"]
            alpha_pred[trimap == 0] = 0.0
            alpha_pred[trimap == 2] = 1.0

            alpha_pred = alpha_pred[0, 0, ...].data.cpu().numpy() * 255
            interm.update({"step_"+str(counter).zfill(3): alpha_pred})
            return interm
        else:
            return unnormalize_img(noised_pha / self.scale)

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape, 
        condition,
        noise=None,
        device=None,
        sample_list = None,
        GTalpha=None
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        # trimap = condition["trimap"]
        batch = shape[0]

        if noise is not None:
            noised_pha = noise
        else:
            noised_pha = torch.randn(*shape, device=device)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        condition["features"] = None
        counter = 0
        interm = {}
        for times, times_next in tqdm(time_pairs, desc = 'ddim sampling loop time step'):

            # get times and noise levels

            gamma = self.gamma_schedule(times)
            gamma_next = self.gamma_schedule(times_next)

            padded_gamma, padded_gamma_next = map(partial(right_pad_dims_to, noised_pha), (gamma, gamma_next))

            alpha, sigma = gamma_to_alpha_sigma(padded_gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(padded_gamma_next, self.scale)

            # add the time delay

            times_next = (times_next - self.time_difference).clamp(min = 0.)

            model_input = condition
            maybe_normalized_pha = self.maybe_normalize_img_variance(noised_pha)

            # maybe_normalized_pha[trimap == 0] = 0
            # maybe_normalized_pha[trimap == 1] = 1

            model_input.update({"x_t":maybe_normalized_pha})
            model_input.update({"timestep":times})

            # predict x0
            if self.model_mean_type == ModelMeanType.START_X:
                # model_output = model(input)["phas"]
                model_output = model(model_input, condition["features"])
                pred, features = model_output["phas"], model_output["feature"]
            elif self.model_mean_type == ModelMeanType.EPSILON:
                # model_output = model(input)["noise"]
                model_output = model(model_input, condition["features"])
                pred, features = model_output["noise"], model_output["feature"]
            condition["features"] = features # 下一个step重复利用features

            # calculate x0 and noise
            if self.model_mean_type == ModelMeanType.START_X: #case2 model预测x0
                pred_xstart = pred
            else: #case3 预测eps的期望值
                pred_xstart = safe_div(noised_pha - sigma * pred, alpha)

            # get predicted noise
            # clip x0
            pred_xstart.clamp_(-1., 1.)
            if GTalpha == None:
                pred_noise = safe_div(noised_pha - alpha * pred_xstart, sigma)
            else:
                GTpha = normalize_img(GTalpha)
                pred_noise = safe_div(noised_pha - alpha * GTpha, sigma)

            if sample_list:
                if counter in sample_list:
                    alpha_pred = unnormalize_img(pred_xstart)
                    alpha_pred = torch.clamp(alpha_pred, 0., 1.)
                    trimap = condition["trimap"]
                    alpha_pred[trimap == 0] = 0.0
                    alpha_pred[trimap == 2] = 1.0

                    alpha_pred = alpha_pred[0, 0, ...].data.cpu().numpy() * 255
                    interm.update({"step_"+str(counter).zfill(3): alpha_pred})
                if counter == sample_list[-1]:
                    return interm
            counter += 1
            # calculate x next
            if GTalpha == None:
                noised_pha = pred_xstart * alpha_next + pred_noise * sigma_next
            else:
                GTpha = normalize_img(GTalpha)
                noised_pha = GTpha * alpha_next + pred_noise * sigma_next
        if sample_list != None and (counter == sample_list[-1] or counter == len(time_pairs)):
            pred_xstart = unnormalize_img(noised_pha / self.scale)
            alpha_pred = torch.clamp(pred_xstart, 0., 1.)
            trimap = condition["trimap"]
            alpha_pred[trimap == 0] = 0.0
            alpha_pred[trimap == 2] = 1.0

            alpha_pred = alpha_pred[0, 0, ...].data.cpu().numpy() * 255
            interm.update({"step_"+str(counter).zfill(3): alpha_pred})
            return interm
        else:
            return unnormalize_img(noised_pha / self.scale)



    def training_losses(self, model, input, model_kwargs=None, noise=None, self_align_stage1=False, self_align_stage2=False):
        images, trimap, pha = input['image'], input['trimap'], input['alpha']
        fg, bg = input['fg'], input['bg']

        trimap[trimap < 85] = 0
        trimap[trimap >= 170] = 1
        trimap[trimap >= 85] = 0.5

        batch, c, h, w = pha.shape
        device = pha.device
        
        pha = normalize_img(pha)

        if noise is None:
            noise = torch.randn_like(pha)

        input["trimap"] = trimap

        if self_align_stage1:
            # print("self_align_stage1")
            temp_times = torch.ones((batch,), device=device).float()
            times = torch.zeros((batch,), device = device).float().uniform_(0, 1.)
            noised_pha_temp = noise
        elif self_align_stage2:
            # print("self_align_stage2")
            difference = torch.zeros((batch,), device = device).float().uniform_(0, 1.)
            times = torch.zeros((batch,), device = device).float().uniform_(0, 1.) * (1 - difference)

            # times = torch.zeros((batch,), device = device).float().uniform_(0, 1.)
            # difference = torch.zeros((batch,), device = device).float().uniform_(0, 1.) * (1 - times)
            temp_times = times + difference
            gamma_temp = self.gamma_schedule(temp_times)
            padded_gamma_temp = right_pad_dims_to(pha, gamma_temp)
            alpha_temp, sigma_temp = gamma_to_alpha_sigma(padded_gamma_temp, self.scale)
            noised_pha_temp = alpha_temp * pha + sigma_temp * noise
        else:
            temp_times = None
            times = torch.zeros((batch,), device = device).float().uniform_(0, 1.)
        
        gamma = self.gamma_schedule(times)
        padded_gamma = right_pad_dims_to(pha, gamma)
        alpha, sigma =  gamma_to_alpha_sigma(padded_gamma, self.scale)

        pha_pred = None
        if temp_times != None:
            maybe_normalized_pha_temp = self.maybe_normalize_img_variance(noised_pha_temp)
            input.update({"timestep":temp_times})
            input.update({"x_t":maybe_normalized_pha_temp})

            features = model(input, intermidate_out=True)['feature']
            if self.model_mean_type == ModelMeanType.START_X:
                pha_pred = model(input, features=features)['phas']
            elif self.model_mean_type == ModelMeanType.EPSILON:
                noise_pred = model(input, features=features)["noise"]
                gamma_temp = self.gamma_schedule(temp_times)
                padded_gamma_temp = right_pad_dims_to(pha, gamma_temp)
                alpha_temp, sigma_temp = gamma_to_alpha_sigma(padded_gamma_temp, self.scale)

                pha_pred = safe_div(noise - sigma_temp * noise_pred, alpha_temp)

            noised_pha = alpha * pha_pred.detach() + sigma * noise
        else:
            noised_pha = alpha * pha + sigma * noise
            features = None
        
        
        maybe_normalized_pha = self.maybe_normalize_img_variance(noised_pha)

        input.update({"timestep":times})
        input.update({"x_t":maybe_normalized_pha})

        if self.model_mean_type == ModelMeanType.START_X:
            model_output = model(input, features=features)["phas"]
        elif self.model_mean_type == ModelMeanType.EPSILON:
            model_output = model(input, features=features)["noise"]

        targets = {
            ModelMeanType.START_X: pha,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == targets.shape == pha.shape

        tensorboard_images = {}
        sample_map = get_unknown_tensor(trimap*2)
        if self.model_mean_type == ModelMeanType.START_X:
            pred_alpha = model_output
            tensorboard_images['x_start_pred_by_Unet'] = unnormalize_img(model_output)
        elif self.model_mean_type == ModelMeanType.EPSILON:
            # pred_alpha = self._predict_xstart_from_eps(x_t, t, model_output)
            pred_alpha = safe_div(noised_pha - sigma * model_output, alpha)
            tensorboard_images['noise'] = noise
            tensorboard_images['pred_noise'] = model_output
            tensorboard_images['x_start_pred_by_eps'] = unnormalize_img(pred_alpha)
        else:
            raise ValueError(f'invalid 387 line model_mean_type {self.model_mean_type}')
        
        tensorboard_images['x_t'] = noised_pha
        

        terms = self.criterion(sample_map, model_output, pred_alpha, targets, pha, fg, bg, images, pha_pred)

        return terms, tensorboard_images, times