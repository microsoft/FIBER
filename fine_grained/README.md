# FIBER fine-grained stage


## Introduction
Here we provide the code for the fine-grained training (both pre-training and downstream fine-tuning) of FIBER.  
Using our two-stage pre-training, FIBER-Base is able to leverage image-text data to outperform previous state-of-the-art with magnitudes less box annotated data. 
We set new state of the art on Flickr30k entities for phrase grounding, on Referring Expression Comprehension as well as on Object Detection benchmarks. 

We provide code for:

1. **pre-training** FIBER on detection and grounding data;
2. **zero-shot evaluating** FIBER on standard benchmarks (COCO, LVIS, Flickr30K) and custom COCO-formated datasets;
3. **fine-tuning** FIBER on standard benchmarks (COCO, LVIS) and custom COCO-formated datasets;
4. **fine-tuning** FIBER on Referring expression comprehension (RefCOCO, RefCOCO+ and RefCOCOg)

Please see respective sections for instructions.

## Installation and Setup

***Environment***
This repo requires Pytorch>=1.9 and torchvision. We recommend using docker to setup the environment. You can use this pre-built docker image ``docker pull pengchuanzhang/maskrcnn:ubuntu18-py3.7-cuda10.2-pytorch1.9`` or this one ``docker pull pengchuanzhang/pytorch:ubuntu20.04_torch1.9-cuda11.3-nccl2.9.9`` depending on your GPU.

Then install the following packages:
```
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo pytorch-lightning
pip install transformers 
python setup.py build develop --user
```

## Pre-Training

***Required Data.***  Prepare ``Objects365``, ``Flickr30K``, and ``MixedGrounding`` data as in [DATA.md](DATA.md). 

***Command for reproducing the pre-training (config file provided in model zoo below).***

```
python -m torch.distributed.launch --nnodes 4 --nproc_per_node=16 tools/train_net.py \
    --config-file configs/pretrain/mixed_nococo_flickr_objects365.yaml \
    --skip-test --use-tensorboard --override_output_dir {output_dir} \
    MODEL.WEIGHT {path to downloaded weights}
```

We use 4 nodes, 16 gpus of V100 32GB GPUs to do the training. You will be required to set up the environment according to your machine/cluster.

**Model Zoo (Pre-trained)**

|Model | Coarse-grained data   | Weight used to initialize|  Pre-train data | Config | Final pre-trained Weight |
|---------- |---------- |---------- |---------- |---------- |---------- |
|FIBER-B | COCO, VG, SBU, GCC | [weight](https://datarelease.blob.core.windows.net/fiber/coarse_grained/fiber_coarse_init.pt) | Flickr30k, MixedNoCOCO, Objects365 | [config](configs/pretrain/mixed_nococo_flickr_objects365.yaml) | [weight](https://datarelease.blob.core.windows.net/fiber/fine_grained/fiber_coarse_then_fine.pth) |
|FIBER-B | COCO, VG, SBU, GCC | [weight](https://datarelease.blob.core.windows.net/fiber/coarse_grained/fiber_coarse_init_for_refexp.pt) |Flickr30k, MixedNoCOCO (cleaned for Refexp) , Objects365 | [config](configs/pretrain/mixed_nococo_flickr_objects365_refexpclean.yaml) | [weight](https://datarelease.blob.core.windows.net/fiber/fine_grained/fiber_coarse_then_fine_for_refexp.pth) |



## (Zero-Shot) Evaluation

### COCO Evaluation

Prepare ``COCO/val2017`` data as in [DATA.md](DATA.md). Set ``{config_file}``, ``{model_checkpoint}`` according to the ``Model Zoo``; set ``{output_dir}`` to a folder where the evaluation results will be stored.

```
python tools/test_grounding_net.py --config-file {config_file} --weight {model_checkpoint} \
        TEST.IMS_PER_BATCH 1 \
        TEST.EVAL_TASK detection \
        OUTPUT_DIR {output_dir}
```

### LVIS Evaluation

We follow MDETR to evaluate with the [FixedAP](https://arxiv.org/pdf/2102.01066.pdf) criterion. Set ``{config_file}``, ``{model_checkpoint}`` according to the ``Model Zoo``. Prepare ``COCO/val2017`` data as in [DATA.md](DATA.md).

```
python -m torch.distributed.launch --nproc_per_node=4 \
        tools/test_grounding_net.py \
        --config-file {config_file} \
        --task_config configs/lvis/minival.yaml \
        --weight {model_checkpoint} \
        TEST.EVAL_TASK detection OUTPUT_DIR {output_dir} 
        TEST.CHUNKED_EVALUATION 40  TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 4 TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM 3000 MODEL.RETINANET.DETECTIONS_PER_IMG 300 MODEL.FCOS.DETECTIONS_PER_IMG 300 MODEL.ATSS.DETECTIONS_PER_IMG 300 MODEL.ROI_HEADS.DETECTIONS_PER_IMG 300
```
If you wish to evaluate on LVIS Val 1.0, set ``--task_config`` to ``configs/lvis/val.yaml``.


### ODinW / Custom Dataset Evaluation

FIBER also supports easy evaluation on a custom dataset, but it must be converted to the [COCO format](https://cocodataset.org/#format-data)

We will use the [Aquarium](https://public.roboflow.com/object-detection/aquarium) dataset from ODinW as an example to show how to evaluate on a custom COCO-formatted dataset.

1. Download the raw dataset from RoboFlow in the COCO format into ``DATASET/odinw/Aquarium``. Each train/val/test split has a corresponding ``annotation`` file and a ``image`` folder. 

2. Remove the background class from the annotation file. This can be as simple as open "_annotations.coco.json" and remove the entry with "id:0" from "categories". For convenience, we provide the modified annotation files for  Aquarium:
    ```
    wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/odinw/Aquarium/Aquarium%20Combined.v2-raw-1024.coco/test/annotations_without_background.json -O DATASET/odinw/Aquarium/Aquarium\ Combined.v2-raw-1024.coco/test/annotations_without_background.json
    wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/odinw/Aquarium/Aquarium%20Combined.v2-raw-1024.coco/train/annotations_without_background.json -O DATASET/odinw/Aquarium/Aquarium\ Combined.v2-raw-1024.coco/train/annotations_without_background.json
    wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/odinw/Aquarium/Aquarium%20Combined.v2-raw-1024.coco/valid/annotations_without_background.json -O DATASET/odinw/Aquarium/Aquarium\ Combined.v2-raw-1024.coco/valid/annotations_without_background.json
    ```
    
4. Then create a yaml file as in ``configs/odinw/Aquarium_Aquarium_Combined.v2-raw-1024.coco.yaml``. A few fields to be noted in the yamls:

    DATASET.CAPTION_PROMPT allows manually changing the prompt (the default prompt is simply concatenating all the categories);

    MODELS.\*.NUM_CLASSES need to be set to the number of categories in the dataset (including the background class). E.g., Aquarium has 7 non-background categories thus MODELS.\*.NUM_CLASSES is set to 8;

4. Run the following command to evaluate on the dataset. Set ``{config_file}``, ``{model_checkpoint}`` according to the ``Model Zoo``. Set {odinw_configs} to the path of the task yaml file we just prepared.

```
python tools/test_grounding_net.py --config-file {config_file} --weight {model_checkpoint} \
      --task_config {odinw_configs} \
      TEST.IMS_PER_BATCH 1 SOLVER.IMS_PER_BATCH 1 \
      TEST.EVAL_TASK detection \
      DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
      DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE False \
      DATASETS.USE_OVERRIDE_CATEGORY True \
      DATASETS.USE_CAPTION_PROMPT True
```

### Flickr30K Evaluation
Prepare ``Flickr30K`` data as in [DATA.md](DATA.md). Set ``{config_file}``, ``{model_checkpoint}`` according to the ``Model Zoo``.

```
python tools/test_grounding_net.py \
        --config-file {config_file} \
        --task_config configs/flickr/test.yaml,configs/flickr/val.yaml \
        --weight {model_checkpoint} \
        OUTPUT_DIR {output_dir} TEST.IMS_PER_BATCH 1 SOLVER.IMS_PER_BATCH 1 TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM 100 TEST.EVAL_TASK grounding
```


## Fine-Tuning

**Model Zoo (Fine-tuned COCO)**

|Model | AP | Config | Best Checkpoint|
| ----------- | ----------- |---------- |---------- |
|FIBER-B | 58.40 | [config](configs/e2e_dyhead_SwinT_B_FPN_coco_finetuning_fusion_backbone.yaml) | [weight](https://datarelease.blob.core.windows.net/fiber/fine_grained/fiber_coco.pth)|

### COCO Fine-Tuning
Prepare the ``COCO`` data as in [DATA.md](DATA.md). Set ``{config_file}``, ``{model_checkpoint}`` according to the ``Model Zoo``.

Command to run finetuning on COCO: 
```
python -m torch.distributed.launch --nproc_per_node=16 tools/train_net.py \
       --config-file {config_file} \
       --skip-test \
       MODEL.WEIGHT {model_checkpoint} \
   
```

For evaluation, please follow the instructions in ``COCO Evaluation``

### LVIS Finetuning 

**Model Zoo (Fine-tuned LVIS)**

|Model | AP | Config | Best Checkpoint| 
|---------- |---------- |---------- |---------- |
|FIBER-B | 56.9 | [config](configs/e2e_dyhead_SwinT_B_FPN_lvis_finetuning_fusion_backbone.yaml) | [weight](https://datarelease.blob.core.windows.net/fiber/fine_grained/fiber_lvis.pth)|

Prepare the ``COCO`` image data as in [DATA.md](DATA.md), and LVIS annotations as described in the corresponding section in [DATA.md](DATA.md). Set ``{config_file}``, ``{model_checkpoint}`` according to the ``Model Zoo``.

```
python -m torch.distributed.launch --nproc_per_node=16 tools/train_net.py \
       --config-file {config_file} \
       --skip-test \
       MODEL.WEIGHT {model_checkpoint} \
   
```

For evaluation, please follow instructions above for LVIS evaluation. 

### ODinW / Custom Dataset Fine-Tuning
Prepare the dataset as in ``ODinW / Custom Dataset Evaluation``.

#### Full Model Fine-Tuning

For tuning with 1/3/5/10-shot, set {custom_shot_and_epoch_and_general_copy} to "1_200_8", "3_200_4", "5_200_2", "10_200_1", respectively.

```
python -m torch.distributed.launch --nproc_per_node=4 tools/finetune.py \
      --config-file {config_file}  --ft-tasks {configs} --skip-test \
      --custom_shot_and_epoch_and_general_copy {custom_shot_and_epoch_and_general_copy} \
      --evaluate_only_best_on_test --push_both_val_and_test \
      MODEL.WEIGHT {model_checkpoint} \
      SOLVER.USE_AMP True TEST.DURING_TRAINING True TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 4 SOLVER.WEIGHT_DECAY 0.05 TEST.EVAL_TASK detection DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding MODEL.BACKBONE.FREEZE_CONV_BODY_AT 2 MODEL.DYHEAD.USE_CHECKPOINT True SOLVER.FIND_UNUSED_PARAMETERS False SOLVER.TEST_WITH_INFERENCE True SOLVER.USE_AUTOSTEP True DATASETS.USE_OVERRIDE_CATEGORY True SOLVER.SEED 10 DATASETS.SHUFFLE_SEED 3 DATASETS.USE_CAPTION_PROMPT True DATASETS.DISABLE_SHUFFLE True \
      SOLVER.STEP_PATIENCE 3 SOLVER.CHECKPOINT_PER_EPOCH 1.0 SOLVER.AUTO_TERMINATE_PATIENCE 8 SOLVER.MODEL_EMA 0.0 SOLVER.TUNING_HIGHLEVEL_OVERRIDE full
```

For tuning with all the data, set {custom_shot_and_epoch_and_general_copy} to "0_200_1".


#### Referring Expressions Finetuning

**Model Zoo (Fine-tuned RefCOCO, RefCOCO+, RefCOCOg)**

|Model | Task |val | testA | testB | Config | Best checkpoint|
|---------- |---------- |---------- |---------- |---------- |---------- |---------- |
|FIBER-B | RefCOCO| 90.68 |  92.59 | 87.26 | [refcoco config](configs/refcoco.yaml) | [weight](https://datarelease.blob.core.windows.net/fiber/fine_grained/fiber_refcoco.pth)|
|FIBER-B | RefCOCO+ | 85.74 | 90.13 | 79.38 | [refcoco+ config](configs/refcocoplus.yaml) | [weight](https://datarelease.blob.core.windows.net/fiber/fine_grained/fiber_refcoco%2B.pth)|
|FIBER-B | RefCOCOg | 87.11 | 87.32 | N/A | [refcocog_config](configs/refcocog.yaml) | [weight](https://datarelease.blob.core.windows.net/fiber/fine_grained/fiber_refcocog.pth) |


Command to run finetuning : 

```
python -m torch.distributed.launch --nproc_per_node=16 tools/train_net.py
        --config-file {corresponding config from model zoo}
        --skip-test
        MODEL.WEIGHT {Path to pretrained checkpoint}
         
```

Evaluation of Refexp models: 

The config file is the one for the model that you are evaluating, and can be found in the RefExp model zoo above, the task config is used to specify the test set. Choose the corresponding test set (TestA/TestB/Test), found in [refexp test configs](configs/refexp). Output dir is the folder where you want to write results. 

Command: 
```
python tools/test_grounding_net.py --config-file {config-file}
        --task_config configs/refexp/{test_set}.yaml
        --weight {model-checkpoint}
        OUTPUT_DIR {output_dir}
        TEST.IMS_PER_BATCH 1 
        SOLVER.IMS_PER_BATCH 1 
        TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM -1 
        TEST.EVAL_TASK grounding 
        MODEL.ATSS.PRE_NMS_TOP_N 3000
        MODEL.ATSS.DETECTIONS_PER_IMG 100
        MODEL.ATSS.INFERENCE_TH 0.0
```

## Acknowledgement 
This codebase was built upon the GLIP codebase. We thank the authors for making the code available. 
Please also consider citing their paper:
```
@inproceedings{li2021grounded,
      title={Grounded Language-Image Pre-training},
      author={Liunian Harold Li* and Pengchuan Zhang* and Haotian Zhang* and Jianwei Yang and Chunyuan Li and Yiwu Zhong and Lijuan Wang and Lu Yuan and Lei Zhang and Jenq-Neng Hwang and Kai-Wei Chang and Jianfeng Gao},
      year={2022},
      booktitle={CVPR},
}
```
