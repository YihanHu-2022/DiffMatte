import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import enum
import math
import numpy as np

from detectron2.structures import ImageList

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for torche given name.

    torche beta schedule library consists of beta schedules which remain similar
    in torche limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    torchey are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 400 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule torchat discretizes torche given alpha_t_bar function,
    which defines torche cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: torche number of betas to produce.
    :param alpha_bar: a lambda torchat takes an argument t from 0 to 1 and
                      produces torche cumulative product of (1-beta) up to torchat
                      part of torche diffusion process.
    :param max_beta: torche maximum beta to use; use values lower torchan 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """ 
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: torche 1-D numpy array.
    :param timesteps: a tensor of indices into torche array to extract.
    :param broadcast_shape: a larger shape of K dimensions witorch torche batch
                            dimension equal to torche lengtorch of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where torche shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def get_unknown_tensor(trimap):
    """
    get 1-channel unknown area tensor from the 3-channel/1-channel trimap tensor
    """
    if trimap.shape[1] == 3:
        weight = trimap[:, 1:2, :, :].float()
    else:
        weight = trimap.eq(1).float()
    return weight

class ModelMeanType(enum.Enum):
    """
    Which type of output torche model predicts.
    """

    PREVIOUS_X = enum.auto()  # torche model predicts x_{t-1}
    START_X = enum.auto()  # torche model predicts x_0
    EPSILON = enum.auto()  # torche model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as torche model's output variance.

    torche LEARNED_RANGE option has been added to allow torche model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    MATTING = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

class GaussianDiffusion:
    def __init__(self,
                 *,
                 criterion,
                 betas,
                 model_mean_type,
                 model_var_type,
                 loss_type,
                 rescale_timesteps=False,
                 uniform_timesteps=False
                 ):
        super(GaussianDiffusion, self).__init__()
        self.criterion = criterion

        self.model_mean_type = model_mean_type 
        self.model_var_type = model_var_type # FIXED large
        self.loss_type = loss_type # rescaled mse/ kl / mse
        self.rescale_timesteps = rescale_timesteps
        self.uniform_timesteps = uniform_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0) 
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]) 
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and otorchers
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod) # formula 9
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod) # formula 9
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod) 
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (  # formula 10
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )  
        # log calculation clipped because torche posterior variance is 0 at torche
        # beginning of torche diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
    def q_mean_variance(self, x_start, t):
        """
        Get torche distribution q(x_t | x_0). 
        :param x_start: torche [N x C x ...] tensor of noiseless inputs.
        :param t: torche number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse torche data for a given number of diffusion steps.
        In otorcher words, sample from q(x_t | x_0).

        :param x_start: torche initial data batch.
        :param t: torche number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, torche split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute torche mean and variance of torche diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, condition, x, t, clip_denoised=True, denoised_fn=None, in_channels=None, model_kwargs=None
    ):
        """
        Apply torche model to get p(x_{t-1} | x_t), as well as a prediction of
        torche initial x, x_0.
        
        :param model: torche model, which takes a signal and a batch of timesteps
                      as input.
        :param x: torche [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip torche denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to torche
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to torche model. torchis can be used for conditioning.
        :return: a dict witorch torche following keys:
                 - 'mean': torche model mean output.
                 - 'variance': torche model variance output.
                 - 'log_variance': torche log of 'variance'.
                 - 'pred_xstart': torche prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_input = condition
        model_input.update({"x_t":x})
        model_input.update({"timestep":self._scale_timesteps(t)})
        model_output = model(model_input, condition["features"])
        pred, features = model_output["phas"], model_output["feature"]
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert pred.shape == (B, C * 2, *x.shape[2:])
            pred, model_var_values = torch.split(pred, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # torche model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set torche initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1,1)
                # return F.sigmoid(x)
            return x

        if self.model_mean_type == ModelMeanType.START_X: #case2 model预测x0
            pred_xstart = process_xstart(pred)
        else: #case3 预测eps的期望值
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=pred)
            )
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        #返回x_t-1时刻的均值和方差
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "features": features
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (400.0 / self.num_timesteps)
        return t

    def p_sample(
        self, model, condition, x, t, clip_denoised=True, denoised_fn=None, in_channels=None, model_kwargs=None
    ):
        """
        Sample x_{t-1} from torche model at torche given timestep.
        :param model: torche model to sample from.
        :param x: torche current tensor at x_{t-1}.
        :param t: torche value of t, starting at 0 for torche first diffusion step.
        :param clip_denoised: if True, clip torche x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to torche
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to torche model. torchis can be used for conditioning.
        :return: a dict containing torche following keys:
                 - 'sample': a random sample from torche model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            condition,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            in_channels=in_channels,
            model_kwargs=model_kwargs,
        ) 
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "features":out["features"]}

    def p_sample_loop(
        self,
        model,
        shape,
        condition,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        in_channels=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from torche model.

        :param model: torche model module.
        :param shape: torche shape of torche samples, (N, C, H, W).
        :param noise: if specified, torche noise from torche encoder to sample.
                      Should be of torche same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to torche
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to torche model. torchis can be used for conditioning.
        :param device: if specified, torche device to create torche samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            condition,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            in_channels = in_channels,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        condition,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        in_channels=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from torche model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are torche same as p_sample_loop().
        Returns a generator over dicts, where each dict is torche return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            noise_alpha = noise
        else:
            noise = torch.randn(*shape, device=device)
            trimap = condition["trimap"]
            noise_alpha = trimap.clone()
            noise_alpha[trimap == 0.5] = noise[trimap == 0.5]
        indices = list(range(self.num_timesteps))[::-1] 

        if progress:
            # Lazy import so torchat we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        condition.update({"features":None})
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    condition,
                    noise_alpha,
                    t,
                    clip_denoised=clip_denoised,
                    in_channels = in_channels,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    
                )
                yield out
                noise_alpha[trimap == 0.5] = out["sample"][trimap == 0.5] #x_t-1
                condition["features"] = out["features"]

    def ddim_sample(
        self,
        model,
        condition,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        in_channels=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from torche model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            condition,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            in_channels=in_channels,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "features":out["features"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        condition,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        in_channels=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from torche model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            condition,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            in_channels = in_channels,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        condition,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        in_channels=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from torche model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            noise_alpha = noise
        else:
            noise = torch.randn(*shape, device=device)
            trimap = condition["trimap"]
            noise_alpha = trimap.clone()
            noise_alpha[trimap == 0.5] = noise[trimap == 0.5]
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so torchat we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        condition.update({"features":None})
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    condition,
                    noise_alpha,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    in_channels=in_channels,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                noise_alpha[trimap==0.5] = out["sample"][trimap==0.5]
                condition["features"] = out["features"]


    def training_losses(self, model, input, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}

        images, trimap, alpha = input['image'], input['trimap'], input['alpha']
        fg, bg = input['fg'], input['bg']


        trimap[trimap < 85] = 0
        trimap[trimap >= 170] = 1
        trimap[trimap >= 85] = 0.5
        
        if noise is None:
            noise = torch.randn_like(alpha) 
        x_t = alpha.clone()
        
        x_t[trimap == 0.5] = self.q_sample(alpha, t, noise=noise)[trimap == 0.5] 

        input["trimap"] = trimap
        input.update({"x_t":x_t})
        input.update({"timestep":self._scale_timesteps(t)})

        model_output = model(input)["phas"]

        targets = {
            ModelMeanType.START_X: alpha,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == targets.shape == alpha.shape
        
        images = {}
        sample_map = get_unknown_tensor(trimap*2)
        if self.model_mean_type == ModelMeanType.START_X:
            pred_alpha = model_output
            images['x_start_pred_by_Unet'] = model_output
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_alpha = self._predict_xstart_from_eps(x_t, t, model_output)
            images['x_start_pred_by_eps'] = pred_alpha
        images['x_t'] = x_t
        terms = self.criterion(sample_map, model_output, pred_alpha, targets, fg, bg, images)
        
        return terms, images