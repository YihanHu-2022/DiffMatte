#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

from engine import DiffusionMattingTrainer
from modeling.diffusion.resample import create_named_schedule_sampler
from utils.logger import power_default_writers
#running without warnings
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    loss = instantiate(cfg.loss)
    if not cfg.diffusion.uniform_timesteps:
        cfg.diffusion.inference_mode = not(model.training)
    cfg.diffusion.criterion = loss
    diffusion = instantiate(cfg.diffusion)
    cfg.difmatte.model = model
    cfg.difmatte.diffusion = diffusion
    difmatte = instantiate(cfg.difmatte)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(difmatte))
    difmatte.to(cfg.train.device)
    if not cfg.diffusion.uniform_timesteps:
        schedule_sampler = create_named_schedule_sampler(cfg.train.schedule_sampler, diffusion) #两种采样器：均匀采样t & 基于loss来进行重要性采样t    影响训练过程中网络的输入
    else:
        schedule_sampler = None
    

    cfg.optimizer.params.model = difmatte
    optim = instantiate(cfg.optimizer)

    train_dataset = instantiate(cfg.train_dataset)
    cfg.dataloader.train.dataset = train_dataset
    cfg.dataloader.train.sampler.dataset = train_dataset
    train_loader = instantiate(cfg.dataloader.train)

    difmatte = create_ddp_model(difmatte, **cfg.train.ddp)
    trainer = DiffusionMattingTrainer(
        model=difmatte, 
        data_loader=train_loader, 
        optimizer=optim, 
        schedule_sampler=schedule_sampler,
        log_image_period=cfg.train.log_image_period,
        self_align_stage1_step = cfg.train.self_align_stage1_step,
        self_align_stage2_step = cfg.train.self_align_stage2_step
    )
    checkpointer = DetectionCheckpointer(
        difmatte,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, difmatte)),
            hooks.PeriodicWriter(
                power_default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_loss_period
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # args.dist_url = "tcp://127.0.0.1:51337"
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )