train = dict(
    output_dir="./output",
    init_checkpoint="",
    max_iter=90000,
    amp=dict(enabled=False),  # options for Automatic Mixed Precision
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=True,
        find_unused_parameters=False,
        fp16_compression=True,
    ),
    checkpointer=dict(period=10000, max_to_keep=200),  # options for PeriodicCheckpointer
    eval_period=5000,
    log_loss_period=10,
    log_image_period = 2000, #Must be a multiple of log_loss_period
    device="cuda",
    seed=8282,
    schedule_sampler="uniform",
    self_align_stage1_step = 1e8,
    self_align_stage2_step = 1e8,
    # ...
)