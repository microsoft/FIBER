# Coarse-grained Pre-training of FIBER

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Pre-trained Checkpoints

Here are the pre-trained models:
1. FIBER pre-trained on GCC+SBU+COCO+VG [link]()
2. FIBER fine-tuned on COCO IR/TR [link]()
3. FIBER fine-tuned on Flickr30k IR/TR [link]()
4. FIBER fine-tuned on VQAv2 [link]()
5. FIBER fine-tuned on NLVR2 [link]()
6. FIBER fine-tuned on COCO Captioning [link]()


## Dataset Preparation

We follow [ViLT](https://github.com/dandelin/ViLT) and use `pyarrow` to serialize the datasets. See [this link](https://github.com/dandelin/ViLT/blob/master/DATA.md) for details.

## Pre-training

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_pretrain_mlm_itm_itc per_gpu_batchsize=<BS_FITS_YOUR_GPU> image_size=<IMAGE_SIZE>
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=8 task_mlm_itm_itc per_gpu_batchsize=8 image_size=384
``` 

## Fine-tuning on Downstream Tasks

### VQAv2

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_vqa per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> image_size=<IMAGE_SIZE>
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_vqa per_gpu_batchsize=8 load_path=fiber_pretrain.ckpt image_size=576
``` 

### COCO IR/TR

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_irtr_coco get_recall_metric=False per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> image_size=<IMAGE_SIZE>
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=16 task_finetune_irtr_coco get_recall_metric=False per_gpu_batchsize=8 load_path=fiber_pretrain.ckpt image_size=576
``` 

### Flickr30k IR/TR

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_irtr_f30k get_recall_metric=False per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> image_size=<IMAGE_SIZE>
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=16 task_finetune_irtr_f30k get_recall_metric=False per_gpu_batchsize=8 load_path=fiber_coco_irtr.ckpt image_size=576
``` 

### NLVR2

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES>  task_finetune_nlvr2 per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> image_size=<IMAGE_SIZE>
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1  task_finetune_nlvr2 per_gpu_batchsize=8 load_path=fiber_pretrain.ckpt image_size=384
``` 

### COCO Captioning

```bash
TODO
```

## Evaluation on Downstream Tasks

### VQAv2

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_vqa per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> image_size=<IMAGE_SIZE> test_only=True
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_vqa per_gpu_batchsize=32 load_path=fiber_vqa.ckpt image_size=576 test_only=True
``` 

Then, submit the json file in the `result` directory to [eval.ai](https://eval.ai/web/challenges/challenge-page/830/overview) evaluation server to get the test-dev and/or test-std scores.

### COCO IR/TR

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_irtr_coco get_recall_metric=True per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> image_size=<IMAGE_SIZE> test_only=True
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_coco get_recall_metric=True per_gpu_batchsize=32 load_path=fiber_coco_irtr.ckpt image_size=576 test_only=True
``` 

The returned values are IR R@1, R@5, R@10 and TR R@1, R@5, R@10.


### Flickr30k IR/TR

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_irtr_f30k get_recall_metric=True per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> image_size=<IMAGE_SIZE> test_only=True
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_f30k get_recall_metric=True per_gpu_batchsize=32 load_path=fiber_f30k_irtr.ckpt image_size=576 test_only=True
``` 

The returned values are IR R@1, R@5, R@10 and TR R@1, R@5, R@10.

### NLVR2

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES>  task_finetune_nlvr2 per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> image_size=<IMAGE_SIZE> test_only=True
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1  task_finetune_nlvr2 per_gpu_batchsize=32 load_path=fiber_nlvr2.ckpt image_size=384 test_only=True
``` 

### COCO Captioning

```bash
TODO
```

## Acknowledgements

The code is based on [METER](https://github.com/zdou0830/METER) and [ViLT](https://github.com/dandelin/ViLT) licensed under [Apache 2.0](https://github.com/dandelin/ViLT/blob/master/LICENSE) and some of the code is borrowed from [CLIP](https://github.com/openai/CLIP) and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).
