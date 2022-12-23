import torch
import random

from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from .dist_utils import all_gather
from .objectives import compute_itc_recall, compute_itm_recall
from ..gadgets.my_metrics import Accuracy, VQAScore, Scalar


def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v <= 0:
                continue
            if k == "vqa":
                setattr(pl_module, f"{split}_{k}_score", VQAScore())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "vae":
                setattr(pl_module, f"{split}_{k}_score", VQAScore())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_kld", Scalar())
            elif k == "encoder_kl":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_kld_x", Scalar())
                setattr(pl_module, f"{split}_{k}_kld_xy", Scalar())
            elif k == "inference_vae":
                setattr(pl_module, f"conditional_logp", Scalar())
                setattr(pl_module, f"interventional_logp", Scalar())
            elif k == "nlvr2":
                if split == "train":
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                else:
                    setattr(pl_module, f"dev_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"dev_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"test_{k}_loss", Scalar())
            elif k == "itm":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "itc":
                setattr(pl_module, f"{split}_{k}_i2t_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_t2i_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_logit_scale", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())


def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    the_metric = 0

    if pl_module.hparams.config["get_recall_metric"] and not pl_module.training:
        if pl_module.hparams.config["get_recall_metric_itc"]:
            (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_itc_recall(pl_module)
        else:
            (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_itm_recall(pl_module)
        print((ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10), pl_module.global_step)
        pl_module.logger.experiment.add_scalar("recalls/ir_r1", ir_r1, pl_module.global_step)
        pl_module.logger.experiment.add_scalar("recalls/ir_r5", ir_r5, pl_module.global_step)
        pl_module.logger.experiment.add_scalar("recalls/ir_r10", ir_r10, pl_module.global_step)
        pl_module.logger.experiment.add_scalar("recalls/tr_r1", tr_r1, pl_module.global_step)
        pl_module.logger.experiment.add_scalar("recalls/tr_r5", tr_r5, pl_module.global_step)
        pl_module.logger.experiment.add_scalar("recalls/tr_r10", tr_r10, pl_module.global_step)
        the_metric += ir_r1.item() + tr_r1.item()

    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v <= 0:
            continue

        value = 0

        if loss_name == "vqa":
            score_metric = getattr(pl_module, f"{phase}_{loss_name}_score")
            value = score_metric.compute()
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            score_metric.reset()

            loss_metric = getattr(pl_module, f"{phase}_{loss_name}_loss")
            pl_module.log(f"{loss_name}/{phase}/loss_epoch", loss_metric.compute())
            loss_metric.reset()
        elif loss_name == "vae":
            score_metric = getattr(pl_module, f"{phase}_{loss_name}_score")
            value = score_metric.compute()
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            score_metric.reset()

            loss_metric = getattr(pl_module, f"{phase}_{loss_name}_loss")
            pl_module.log(f"{loss_name}/{phase}/loss_epoch", loss_metric.compute())
            loss_metric.reset()

            kld_metric = getattr(pl_module, f"{phase}_{loss_name}_kld")
            pl_module.log(f"{loss_name}/{phase}/kld_epoch", kld_metric.compute())
            kld_metric.reset()
        elif loss_name == "encoder_kl":
            loss_metric = getattr(pl_module, f"{phase}_{loss_name}_loss")
            value = loss_metric.compute()
            pl_module.log(f"{loss_name}/{phase}/loss_epoch", value)
            loss_metric.reset()

            kld_x_metric = getattr(pl_module, f"{phase}_{loss_name}_kld_x")
            pl_module.log(f"{loss_name}/{phase}/kld_x_epoch", kld_x_metric.compute())
            kld_x_metric.reset()

            kld_xy_metric = getattr(pl_module, f"{phase}_{loss_name}_kld_xy")
            pl_module.log(f"{loss_name}/{phase}/kld_xy_epoch", kld_xy_metric.compute())
            kld_xy_metric.reset()
        elif loss_name == "inference_vae":
            conditional_logp_metric = getattr(pl_module, "conditional_logp")
            pl_module.log(f"{loss_name}/conditional_logp", conditional_logp_metric.compute())
            conditional_logp_metric.reset()

            interventional_logp_metric = getattr(pl_module, "interventional_logp")
            pl_module.log(f"{loss_name}/interventional_logp", interventional_logp_metric.compute())
            interventional_logp_metric.reset()
        elif loss_name == "nlvr2":
            if phase == "train":
                value = getattr(pl_module, f"train_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/train/accuracy_epoch", value)
                getattr(pl_module, f"train_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/train/loss_epoch",
                    getattr(pl_module, f"train_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"train_{loss_name}_loss").reset()
            else:
                value = getattr(pl_module, f"test_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/test/accuracy_epoch", value)
                getattr(pl_module, f"test_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/test/loss_epoch",
                    getattr(pl_module, f"test_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"test_{loss_name}_loss").reset()

                value = getattr(pl_module, f"dev_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/dev/accuracy_epoch", value)
                getattr(pl_module, f"dev_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/dev/loss_epoch",
                    getattr(pl_module, f"dev_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"dev_{loss_name}_loss").reset()
        elif loss_name == "itm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "itc":
            value = getattr(pl_module, f"{phase}_{loss_name}_i2t_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/i2t_accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_i2t_accuracy").reset()

            value = getattr(pl_module, f"{phase}_{loss_name}_t2i_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/t2i_accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_t2i_accuracy").reset()

            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        else:
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric)


def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [k for k, v in pl_module.hparams.config["loss_names"].items() if v > 0]
    return


def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    if pl_module.hparams.config["exp_name"] in ["finetune_vqa", "finetune_vae"]:
        return torch.optim.Adam(pl_module.vqa_classifier.parameters(), lr=lr)
    elif pl_module.hparams.config["exp_name"] == "posterior_kl":
        return torch.optim.Adam(pl_module.vqa_classifier.encoder_x.parameters(), lr=lr)
    else:
        no_decay = [
            "bias",
            "LayerNorm.bias",
            "LayerNorm.weight",
            "norm.bias",
            "norm.weight",
            "norm1.bias",
            "norm1.weight",
            "norm2.bias",
            "norm2.weight",
        ]
        head_names = ["vqa_classifier", "nlvr2_classifier", "mlm_score", "itm_score", "snli_classifier"]
        cross_modal_names = ["cross_modal", "i2t", "t2i"]
        lr_mult_head = pl_module.hparams.config["lr_mult_head"]
        lr_mult_cross_modal = pl_module.hparams.config["lr_mult_cross_modal"]
        end_lr = pl_module.hparams.config["end_lr"]
        decay_power = pl_module.hparams.config["decay_power"]
        optim_type = pl_module.hparams.config["optim_type"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in pl_module.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and not any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": wd,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in pl_module.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and not any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in pl_module.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(bb in n for bb in head_names)
                    and not any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": wd,
                "lr": lr * lr_mult_head,
            },
            {
                "params": [
                    p
                    for n, p in pl_module.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and any(bb in n for bb in head_names)
                    and not any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": 0.0,
                "lr": lr * lr_mult_head,
            },
            {
                "params": [
                    p
                    for n, p in pl_module.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": wd,
                "lr": lr * lr_mult_cross_modal,
            },
            {
                "params": [
                    p
                    for n, p in pl_module.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": 0.0,
                "lr": lr * lr_mult_cross_modal,
            },
        ]

        if optim_type == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))
        elif optim_type == "adam":
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
        elif optim_type == "sgd":
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

        if pl_module.trainer.max_steps is None:
            max_steps = (
                len(pl_module.trainer.datamodule.train_dataloader())
                * pl_module.trainer.max_epochs
                // pl_module.trainer.accumulate_grad_batches
            )
        else:
            max_steps = pl_module.trainer.max_steps

        warmup_steps = pl_module.hparams.config["warmup_steps"]
        if isinstance(pl_module.hparams.config["warmup_steps"], float):
            warmup_steps = int(max_steps * warmup_steps)

        if decay_power == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
            )
        else:
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
                lr_end=end_lr,
                power=decay_power,
            )

        sched = {"scheduler": scheduler, "interval": "step"}

        return (
            [optimizer],
            [sched],
        )