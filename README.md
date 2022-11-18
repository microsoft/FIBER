# Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone

<img src="figs/fiber_pipeline.png" width="800">

[Website](https://ashkamath.github.io/FIBER_page/) • [Paper](https://arxiv.org/abs/2206.07643)

## Introduction

We present **FIBER (Fusion In-the-Backbone transformER)** a novel Vision and Language architecture that performs deep multi-modal fusion. We also propose a new Vision-Language Pre-training (VLP) strategy, that first learns through coarse-grained image level objectives, and then obtains better fine-grained understanding capabilties by training on image-text-box data. While previous work required pseudo-annotating large amounts of image-text data to boost performance on fine-grained reasoning tasks, we show that we can equal and often surpass these results using our two-stage approach, using 25x less box annotated data. This opens the doors to scale up fine-grained models in an efficient manner without resorting to high resolution training using box annotated data. Our improved architecture also obtains state of the art performance on VQAv2, NLVR2, COCO captioning and Image-text Retrieval while being more efficient in terms of training time and memory than existing coarse and fine-grained models having similar performance. 

**TL;DR**
* What : A new architecture for Vision and Language tasks + a new pre-training strategy that benefits both image level and region level tasks. 
* How: We add cross-modality attention blocks into the image and text backbone & split pre-training into low and high resolution stages.
* Outcome: State-of-the art results on a captioning, VQA, NLVR2, and more + efficient use of expensive fine-grained data, surpassing phrase grounding performance of models using 25x more box-annotated data!

In this repository we provide code and pre-trained checkpoints for **coarse-grained** pre-training on image-text data and **fine-grained** pre-training on image-text-box data. We also provide instructions, code and checkpoints for fine-tuning FIBER on all the downstream tasks reported in the paper. Please see respective directories for instructions.



## Model Performance

### Using 1st stage pre-training 

**Results on Visual Question Answering, Visual Reasoning, Image-Text Retrieval and Image Captioning**
<table border="1" width="100%">
    <tr align="center">
        <th>Task</th><th>VQAv2</th><th>NLVR2</th><th>F30k Retrieval</th><th>COCO Retrieval</th><th>COCO Captioning</th>
    </tr>
    <tr align="center">
        <td>Split</td><td>test-std</td><td>test-P</td><td>test</td><td>Karpathy test</td><td>Karpathy test</td>
    </tr>
    <tr align="center">
        <td>Metric</td><td>VQA Score</td><td>Acc.</td><td>IR@1/TR@1</td><td>IR@1/TR@1</td><td>CIDEr</td>
    </tr>
    <tr align="center">
        <td>FIBER-Base</td><td>78.46</td><td>85.52</td><td>81.44/92.90 (ITC) 84.10/95.10 (ITM)</td><td>58.01/75.38 (ITC) 59.03/75.14 (ITM)</td><td>144.4</td>
    </tr>
</table>

### Using 2nd stage pre-training

**Results on Phrase Grounding and Referring Expression Comprehension**
<table border="1" width="100%">
    <tr align="center">
        <th>Task</th><th>F30k Grounding</th><th>RefCOCO</th><th>RefCOCO+</th><th>RefCOCOg</th>
    </tr>
    <tr align="center">
        <td>Split</td><td>test</td><td>val/testA/testB</td><td>val/testA/testB</td><td>val/test</td>
    </tr>
    <tr align="center">
        <td>Metric</td><td>R@1/R@5/R@10</td><td>Acc.</td><td>Acc.</td><td>Acc.</td>
    </tr>
    <tr align="center">
        <td>FIBER-Base</td><td>87.4/96.4/97.6</td><td>90.68/92.59/87.26</td><td>85.74/90.13/79.38</td><td>87.11/87.32</td>
    </tr>
</table>

**Results on Object Detection on COCO, LVIS and ODinW**
<table border="1" width="100%">
    <tr align="center">
        <th>Task</th><th>COCO Detection</th><th>LVIS</th><th>ODinW</th>
    </tr>
    <tr align="center">
        <td>Split</td><td>Val 2017</td><td>MiniVal</td><td>13 Datasets</td>
    </tr>
    <tr align="center">
        <td>Metric</td><td>Zero-shot/Fine-tune AP</td><td>Zero-shot/Fine-tune AP</td><td>Avg. Zero-shot/Fine-tune AP</td>
    </tr>
    <tr align="center">
        <td>FIBER-Base</td><td>49.3/58.4</td><td>35.8/56.9</td><td>47.0/65.9</td>
    </tr>
</table>


## Citation
```
@inproceedings{fiber2022,
  title={Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone},
  author={Dou, Zi-Yi* and Kamath, Aishwarya* and Gan, Zhe* and Zhang, Pengchuan and Wang, Jianfeng and Li, Linjie and Liu, Zicheng and Liu, Ce and LeCun, Yann and Peng, Nanyun and Gao, Jianfeng and Wang, Lijuan},
  booktitle={NeurIPS},
  year={2022},
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
