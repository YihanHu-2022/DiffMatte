import modeling.diffusion.gaussian_diffusion as gd
from modeling.diffusion.respace import SpacedDiffusion, space_timesteps
import modeling.diffusion.uniform_gauss as ug
from modeling.diffusion.uniform_gauss import UniformGauss

def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_mse=False,
    use_mat=True,
    predict_xstart=True,
    rescale_timesteps=False,
    timestep_respacing="",
    inference_mode=False,
    criterion=None,
    uniform_timesteps=False
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    loss_type = []
    if use_mse:
        loss_type.append(gd.LossType.MSE)
    if use_mat:
        loss_type.append(gd.LossType.MATTING)
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        uniform_timesteps=uniform_timesteps,
        inference_mode=inference_mode,
        criterion=criterion
    )

def create_uniform_gaussian_diffusion(
    *,
    schedule_kwargs: dict = dict(),
    time_difference = 0.,
    scale = 1.,
    steps=1000,
    noise_schedule="linear",
    predict_xstart=True,
    criterion=None,
    uniform_timesteps=True,
    jump_step=-1
):
    
    return UniformGauss(
        model_mean_type=(
            ug.ModelMeanType.EPSILON if not predict_xstart else ug.ModelMeanType.START_X
        ),
        criterion=criterion,
        timesteps=steps,
        uniform_timesteps = uniform_timesteps,
        noise_schedule=noise_schedule,
        schedule_kwargs=schedule_kwargs,
        time_difference=time_difference,
        scale=scale,
        jump_step=jump_step,
    )