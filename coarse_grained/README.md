# FIBER coarse-grained stage

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Model Checkpoints

Here are the pre-trained models:
1. FIBER pre-trained on GCC+SBU+COCO+VG [link](https://datarelease.blob.core.windows.net/fiber/coarse_grained/fiber_pretrain.ckpt)
2. FIBER fine-tuned on COCO IR/TR with ITC [link](https://datarelease.blob.core.windows.net/fiber/coarse_grained/fiber_coco_irtr_itc.ckpt)
3. FIBER fine-tuned on COCO IR/TR with ITM [link](https://datarelease.blob.core.windows.net/fiber/coarse_grained/fiber_coco_irtr_itm.ckpt)
4. FIBER fine-tuned on Flickr30k IR/TR with ITC [link](https://datarelease.blob.core.windows.net/fiber/coarse_grained/fiber_f30k_irtr_itc.ckpt)
5. FIBER fine-tuned on Flickr30k IR/TR with ITM [link](https://datarelease.blob.core.windows.net/fiber/coarse_grained/fiber_f30k_irtr_itm.ckpt)
6. FIBER fine-tuned on VQAv2 [link](https://datarelease.blob.core.windows.net/fiber/coarse_grained/fiber_vqa.ckpt)
7. FIBER fine-tuned on NLVR2 [link](https://datarelease.blob.core.windows.net/fiber/coarse_grained/fiber_nlvr2.ckpt)
8. FIBER fine-tuned on COCO Captioning [link](https://datarelease.blob.core.windows.net/fiber/coarse_grained/fiber_coco_caption.ckpt)


## Dataset Preparation

We follow [ViLT](https://github.com/dandelin/ViLT) and [METER](https://github.com/zdou0830/METER) to prepare the datasets. See [this link](https://github.com/dandelin/ViLT/blob/master/DATA.md) for details.

## Pre-training

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK

# single-node example
python run.py with data_root=/data2/dsets/dataset num_gpus=64 num_nodes=1 task_mlm_itm_itc per_gpu_batchsize=8

# multi-node on Azure clusters example
python azure_distributed_run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=8 task_mlm_itm_itc per_gpu_batchsize=8
``` 

## Fine-tuning on Downstream Tasks

### VQAv2

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK

# training example
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_vqa per_gpu_batchsize=8 load_path=fiber_pretrain.ckpt

# evaluation example
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_vqa per_gpu_batchsize=32 load_path=fiber_vqa.ckpt test_only=True
# submit the json file in the `result` directory to [eval.ai](https://eval.ai/web/challenges/challenge-page/830/overview) evaluation server to get the test-dev and/or test-std scores.
```

### NLVR2

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK

# training example
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1  task_finetune_nlvr2 per_gpu_batchsize=8 load_path=fiber_pretrain.ckpt

# evaluation example
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1  task_finetune_nlvr2 per_gpu_batchsize=32 load_path=fiber_nlvr2.ckpt test_only=True
``` 

### COCO IR/TR

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK

# itc training example
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_itc_coco per_gpu_batchsize=128 load_path=fiber_pretrain.ckpt

# itc evaluation example
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_itc_coco per_gpu_batchsize=32 load_path=fiber_coco_irtr_itc.ckpt get_recall_metric=True test_only=True

# itm training example
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_itm_coco per_gpu_batchsize=8 load_path=fiber_pretrain.ckpt

# itm evaluation example
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_itm_coco per_gpu_batchsize=32 load_path=fiber_coco_irtr_itc.ckpt get_recall_metric=True get_recall_metric_itc=False test_only=True
``` 

### Flickr30k IR/TR

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK

# itc training example
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_itc_f30k per_gpu_batchsize=128 load_path=fiber_coco_irtr_itc.ckpt

# itc evaluation example
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_itc_f30k per_gpu_batchsize=32 load_path=fiber_f30k_irtr_itc.ckpt get_recall_metric=True test_only=True

# itm training example
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_itm_f30k per_gpu_batchsize=8 load_path=fiber_coco_irtr_itm.ckpt

# itm evaluation example
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_itm_f30k per_gpu_batchsize=32 load_path=fiber_f30k_irtr_itm.ckpt get_recall_metric=True get_recall_metric_itc=False
```

### COCO Captioning

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK

# mle training example
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_caption_mle_coco per_gpu_batchsize=8 load_path=fiber_pretrain.ckpt

# gold training example
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_caption_gold_coco per_gpu_batchsize=8 load_path=fiber_coco_caption_mle.ckpt

# cider optimization training example
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_caption_cider_coco per_gpu_batchsize=8 load_path=fiber_coco_caption_gold.ckpt

# evaluation example
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_caption_mle_coco per_gpu_batchsize=32 load_path=fiber_coco_caption.ckpt
# the generated results will be in `result/caption.json`
```


## Acknowledgements

The code is based on [METER](https://github.com/zdou0830/METER) and [ViLT](https://github.com/dandelin/ViLT) licensed under [Apache 2.0](https://github.com/dandelin/ViLT/blob/master/LICENSE) and some of the code is borrowed from [CLIP](https://github.com/openai/CLIP), [fairseq](https://github.com/facebookresearch/fairseq), [ALBEF](https://github.com/salesforce/ALBEF), and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer). We also thank Wenhui Wang, Li Dong, Furu Wei, Bin Xiao, and Lu Yuan for their helpful discussions.
