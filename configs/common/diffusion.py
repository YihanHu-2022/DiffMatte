from detectron2.config import LazyCall as L
from modeling.diffusion.script_utils import create_gaussian_diffusion, create_uniform_gaussian_diffusion
from modeling.criterion.matting_criterion import DiffusionMattingCriterion

uniform_timesteps = True

if not uniform_timesteps:
    learn_sigma = False
    sigma_small = False
    diffusion_steps = 200
    noise_schedule = "linear"
    timestep_respacing = "ddim10"
    use_mse = True
    use_mat = False
    predict_xstart = True
    rescale_timesteps = True

    criterion = None
    inference_mode=False # No need to change manually, automatically passed in

    diffusion = L(create_gaussian_diffusion)(
            steps=diffusion_steps,
            learn_sigma=learn_sigma,
            sigma_small=sigma_small,
            noise_schedule=noise_schedule,
            use_mse=use_mse,
            use_mat=use_mat,
            predict_xstart=predict_xstart,
            rescale_timesteps=rescale_timesteps,
            timestep_respacing=timestep_respacing,
            inference_mode=inference_mode,
            criterion = criterion,
            uniform_timesteps = uniform_timesteps,
    )
else:
    schedule_kwargs={}
    time_difference = 0.
    scale=1.
    diffusion_steps = 10
    noise_schedule = "linear" #linear, cosine, sigmoid
    predict_xstart = True

    criterion = None

    jump_step = -1

    diffusion = L(create_uniform_gaussian_diffusion)(
            schedule_kwargs = schedule_kwargs,
            time_difference = time_difference,
            scale = scale,
            steps = diffusion_steps,
            uniform_timesteps = uniform_timesteps,
            noise_schedule = noise_schedule,
            predict_xstart = predict_xstart,
            criterion = criterion,
            jump_step = jump_step
    )