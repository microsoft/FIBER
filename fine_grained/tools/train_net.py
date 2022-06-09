# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg, try_to_find
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.metric_logger import MetricLogger, TensorboardLogger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
import numpy as np
import random
import pdb
from maskrcnn_benchmark.utils.amp import autocast, GradScaler


def train(cfg, local_rank, distributed, use_tensorboard=False, debug_nan_checkpoint=None):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if cfg.MODEL.BACKBONE.RESET_BN:
        for name, param in model.named_buffers():
            if "running_mean" in name:
                torch.nn.init.constant_(param, 0)
            if "running_var" in name:
                torch.nn.init.constant_(param, 1)

    if cfg.SOLVER.GRAD_CLIP > 0:
        clip_value = cfg.SOLVER.GRAD_CLIP
        for p in filter(lambda p: p.grad is not None, model.parameters()):
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=0,  # <TODO> Sample data from resume is disabled, due to the conflict with max_epoch
    )

    if cfg.TEST.DURING_TRAINING or cfg.SOLVER.USE_AUTOSTEP:
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
        data_loaders_val = data_loaders_val[0]
    else:
        data_loaders_val = None

    if cfg.MODEL.BACKBONE.FREEZE:
        for p in model.backbone.body.parameters():
            p.requires_grad = False

    if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
        print("LANGUAGE_BACKBONE FROZEN.")
        for p in model.language_backbone.body.parameters():
            p.requires_grad = False

    if cfg.MODEL.FPN.FREEZE:
        for p in model.backbone.fpn.parameters():
            p.requires_grad = False
    if cfg.MODEL.RPN.FREEZE:
        for p in model.rpn.parameters():
            p.requires_grad = False

    # if cfg.SOLVER.PROMPT_PROBING_LEVEL != -1:
    #     if cfg.SOLVER.PROMPT_PROBING_LEVEL == 1:
    #         for p in model.parameters():
    #             p.requires_grad = False

    #         for p in model.language_backbone.body.parameters():
    #             p.requires_grad = True

    #         for name, p in model.named_parameters():
    #             if p.requires_grad:
    #                 print(name, " : Not Frozen")
    #             else:
    #                 print(name, " : Frozen")
    #     else:
    #         assert(0)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=cfg.MODEL.BACKBONE.USE_BN,
            find_unused_parameters=cfg.SOLVER.FIND_UNUSED_PARAMETERS,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, output_dir, save_to_disk)
    extra_checkpoint_data = checkpointer.load(try_to_find(cfg.MODEL.WEIGHT))
    arguments.update(extra_checkpoint_data)

    # For full model finetuning
    # arguments["iteration"] = 0
    # optimizer = make_optimizer(cfg, model)
    # scheduler = make_lr_scheduler(cfg, optimizer)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if use_tensorboard:
        meters = TensorboardLogger(log_dir=cfg.OUTPUT_DIR, start_iter=arguments["iteration"], delimiter="  ")
    else:
        meters = MetricLogger(delimiter="  ")

    if debug_nan_checkpoint is not None:
        debug_nan(cfg, model, debug_nan_checkpoint)

    do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        data_loaders_val,
        meters,
    )

    return model


def debug_nan(cfg, model, checkpoint):
    device = torch.device(cfg.MODEL.DEVICE)
    # device = torch.device("cpu")
    # model.to(device)

    checkpoint = torch.load(checkpoint, map_location="cpu")

    def load_state_dict_flexible(model, state_dict):
        try:
            model.load_state_dict(state_dict)
        except:
            print("Full loading failed!! Try partial loading!!")

        own_state = model.state_dict()

        for name, param in state_dict.items():
            if name not in own_state:
                print("Skipped: " + name)
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
                print("Successfully loaded: " + name)
            except:
                print("Part load failed: " + name)
        print("\n\n")

    load_state_dict_flexible(model, checkpoint["states"])

    images = checkpoint["x"]
    targets = checkpoint["y"]
    captions = checkpoint["captions"]
    positive_map = checkpoint["positive_map"]

    images = images.to(device)
    targets = [i.to(device) for i in targets]
    positive_map = positive_map.to(device)

    """
    On GPU
    """
    import math

    counter = 0
    with torch.no_grad():
        while True:
            print(counter)
            counter += 1
            loss_dict = model(images, targets, captions, positive_map)

            with autocast():
                loss_dict_fp16 = model(images, targets, captions, positive_map)
            losses = sum(loss for loss in loss_dict.values())
            losses_fp_16 = sum(loss for loss in loss_dict_fp16.values())
            loss_value = losses_fp_16.item()
            if not math.isfinite(loss_value):
                pdb.set_trace()

    print(loss_dict)
    print(loss_dict_fp16)

    pdb.set_trace()

    losses = sum(loss for loss in loss_dict.values())

    pdb.set_trace()


"""
def test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=(cfg.MODEL.RPN_ONLY and cfg.MODEL.RPN_ARCHITECTURE == "RPN") or cfg.DATASETS.CLASS_AGNOSTIC,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()
"""


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )

    parser.add_argument(
        "--use-tensorboard",
        dest="use_tensorboard",
        help="Use tensorboardX logger (Requires tensorboardX installed)",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--save_original_config", action="store_true")
    parser.add_argument("--disable_output_distributed", action="store_true")
    parser.add_argument("--debug_nan_checkpoint", default=None)
    parser.add_argument("--override_output_dir", default=None)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        import datetime

        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(0, 7200))

    if args.disable_output_distributed:
        setup_for_distributed(args.local_rank <= 0)

    cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # specify output dir for models
    cfg.OUTPUT_DIR = os.getenv("AMLT_OUTPUT_DIR", cfg.OUTPUT_DIR)
    if args.override_output_dir:
        cfg.OUTPUT_DIR = args.override_output_dir
    cfg.freeze()

    seed = cfg.SOLVER.SEED + args.local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info(args)
    logger.info("Using {} GPUs".format(num_gpus))

    # logger.info("Collecting env info (might take some time)")
    # logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, "config.yml")
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    if args.save_original_config:
        import shutil

        shutil.copy(args.config_file, os.path.join(cfg.OUTPUT_DIR, "config_original.yml"))

    save_config(cfg, output_config_path)

    # Also just save the original config for easy read

    model = train(
        cfg=cfg,
        local_rank=args.local_rank,
        distributed=args.distributed,
        use_tensorboard=args.use_tensorboard,
        debug_nan_checkpoint=args.debug_nan_checkpoint,
    )


if __name__ == "__main__":
    main()
