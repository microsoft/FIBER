from sacred import Experiment

ex = Experiment("FIBER")


def _loss_names(d):
    ret = {
        "itm": 0,
        "itc": 0,
        "mlm": 0,
        "vae": 0,
        "vqa": 0,
        "nlvr2": 0,
        "caption_mle": 0,
        "caption_gold": 0,
        "caption_cider": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "fiber"
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "itc": 1})
    batch_size = (
        4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    )

    # Image settings
    train_transform_keys = ["albef"]
    val_transform_keys = ["albef"]
    image_size = 384
    vit = "swin_base_patch4_window12_384_in22k"
    image_only = False
    draw_false_image = 0
    input_image_embed_size = 1024
    resolution_before = 384
    pretrained_vit = True

    # Text settings
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "roberta-base"
    vocab_size = 50265
    whole_word_masking = False  # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    input_text_embed_size = 768

    # Transformer settings
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    num_fuse_block = 6
    itc_pooler = True  # does not make a difference

    # Optimizer settings
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 100000
    warmup_steps = 10000
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

    # Downstream settings
    get_recall_metric = False
    get_recall_metric_itc = True
    cider_path = None

    # PL Trainer settings
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 8
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 32

    # VAE settings
    latent_size = 512

    # VQA settings
    is_cp = False
    train_subset_ratio = 1


@ex.named_config
def task_pretrain_mlm_itm_itc():
    exp_name = "mlm_itm_itc"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "itc": 1})
    draw_false_image = 0
    batch_size = 4096

    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = False

    learning_rate = 1e-5
    lr_mult_cross_modal = 5
    lr_mult_head = 5
    val_check_interval = 1.0


@ex.named_config
def task_finetune_nlvr2():
    exp_name = "finetune_nlvr2"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    val_check_interval = 1.0
    batch_size = 256
    max_epoch = 5
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 2e-5
    lr_mult_cross_modal = 5
    lr_mult_head = 50
    max_text_len = 50
    train_transform_keys = ["albef_randaug"]
    val_transform_keys = ["albef"]
    image_size = 384
    pretrained_vit = False


@ex.named_config
def task_finetune_vqa():
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    val_check_interval = 1.0
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 0.001
    lr_mult_cross_modal = 5
    lr_mult_head = 50
    max_text_len = 50
    train_transform_keys = ["albef_randaug"]
    val_transform_keys = ["albef"]
    image_size = 576
    pretrained_vit = False


@ex.named_config
def task_finetune_vae():
    exp_name = "finetune_vae"
    datasets = ["vqa"]
    loss_names = _loss_names({"vae": 1})
    val_check_interval = 1.0
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 0.001
    lr_mult_cross_modal = 5
    lr_mult_head = 50
    max_text_len = 50
    train_transform_keys = ["albef_randaug"]
    val_transform_keys = ["albef"]
    image_size = 576
    pretrained_vit = False


@ex.named_config
def task_encoder_kl():
    exp_name = "encoder_kl"
    datasets = ["vqa"]
    loss_names = _loss_names({"encoder_kl": 1})
    val_check_interval = 1.0
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 0.001
    lr_mult_cross_modal = 5
    lr_mult_head = 50
    max_text_len = 50
    train_transform_keys = ["albef_randaug"]
    val_transform_keys = ["albef"]
    image_size = 576
    pretrained_vit = False


@ex.named_config
def task_finetune_irtr_itc_coco():
    exp_name = "finetune_irtr_itc_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"itc": 1})
    batch_size = 1024
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 2e-5
    lr_mult_cross_modal = 5
    lr_mult_head = 5
    max_text_len = 50
    train_transform_keys = ["albef_randaug"]
    val_transform_keys = ["albef"]
    image_size = 576
    pretrained_vit = False


@ex.named_config
def task_finetune_irtr_itc_f30k():
    exp_name = "finetune_irtr_itc_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"itc": 1})
    batch_size = 1024
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 2e-5
    lr_mult_cross_modal = 5
    lr_mult_head = 5
    max_text_len = 50
    train_transform_keys = ["albef_randaug"]
    val_transform_keys = ["albef"]
    image_size = 576
    pretrained_vit = False
    resolution_before = 576


@ex.named_config
def task_finetune_irtr_itm_coco():
    exp_name = "finetune_irtr_itm_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 1})
    draw_false_image = 1
    batch_size = 1024
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 2e-5
    lr_mult_cross_modal = 5
    lr_mult_head = 5
    max_text_len = 50
    train_transform_keys = ["albef_randaug"]
    val_transform_keys = ["albef"]
    image_size = 384
    get_recall_metric_itc = False
    pretrained_vit = False


@ex.named_config
def task_finetune_irtr_itm_f30k():
    exp_name = "finetune_irtr_itm_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"itm": 1})
    draw_false_image = 1
    batch_size = 1024
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 2e-5
    lr_mult_cross_modal = 5
    lr_mult_head = 5
    max_text_len = 50
    train_transform_keys = ["albef_randaug"]
    val_transform_keys = ["albef"]
    image_size = 384
    get_recall_metric_itc = False
    pretrained_vit = False


@ex.named_config
def task_finetune_caption_mle_coco():
    exp_name = "finetune_caption_mle_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"caption_mle": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 5e-5
    lr_mult_cross_modal = 5
    lr_mult_head = 5
    max_text_len = 50
    train_transform_keys = ["albef_randaug"]
    val_transform_keys = ["albef"]
    image_size = 576
    pretrained_vit = False


@ex.named_config
def task_finetune_caption_gold_coco():
    exp_name = "finetune_caption_gold_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"caption_gold": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 1e-5
    lr_mult_cross_modal = 5
    lr_mult_head = 5
    max_text_len = 50
    train_transform_keys = ["albef_randaug"]
    val_transform_keys = ["albef"]
    image_size = 576
    resolution_before = 576
    pretrained_vit = False


@ex.named_config
def task_finetune_caption_cider_coco():
    exp_name = "finetune_caption_cider_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"caption_cider": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 1e-6
    lr_mult_cross_modal = 5
    lr_mult_head = 5
    max_text_len = 50
    train_transform_keys = ["albef_randaug"]
    val_transform_keys = ["albef"]
    image_size = 576
    resolution_before = 576
    pretrained_vit = False
    cider_path = "coco-train-words.p"
