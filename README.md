# LayerCollapse

## Abstract:

Handling the ever-increasing scale of contemporary deep learning and transformer-based models poses a significant challenge. Although great strides have been made in optimizing model compression techniques such as model architecture search and knowledge distillation, the availability of data and computational resources remains a considerable hurdle for these optimizations. This paper introduces LayerCollapse, a novel alternative adaptive model compression methodology. LayerCollapse works by eliminating non-linearities within the network and collapsing two consecutive fully connected layers into a single linear transformation. This approach simultaneously reduces both the number of layers and the parameter count, thereby enhancing model efficiency. We also introduce a compression aware regularizer, which compresses the model in alignment with the dataset quality and model expressiveness, consequently reducing overfitting across tasks. Our results demonstrate LayerCollapse's effective compression and regularization capabilities in multiple fine-grained classification benchmarks, achieving up to 74% post training compression with minimal accuracy loss. We compare this method with knowledge distillation on the same target network, showcasing a five-fold increase in computational efficiency and 8% improvement in overall accuracy on the ImageNet dataset.

# Repository Overview

This GitHub repository contains code implementations and experiments associated with the LayerCollapse methodology, an adaptive model compression technique. The repository is structured as follows:

## Prerequisites:

Before running the code in this repository, ensure that you have installed the required dependencies listed in the `requirements.txt` file. Additionally, download the necessary files from the following link: [Download Files](https://drive.google.com/drive/folders/1UzDey65lFPo2Dle1LP4PcpoT_pvpBJeF?usp=sharing).

## Files:

1. **train.py:**
   - Implementation of the training and evaluation pipeline for a deep learning model, incorporating LayerCollapse methodology with adaptive compression and regularization.
   - Dynamically adjusts hyperparameters during training and provides mechanisms for saving the model and training statistics at specified intervals.

2. **collapsible_mlp.py:**
   - Module providing a flexible and customizable MLP (CollapsibleMlp) with the capability to collapse layers, potentially improving efficiency and reducing model complexity based on specified conditions.

3. **vgg.py:**
   - Classes serving as modular and customizable implementations of VGG-style neural networks.
   - VGG16 class provides a specific configuration commonly used for image classification tasks.

4. **utils.py:**
   - Comprehensive utility for experimenting with different neural network architectures, training configurations, and datasets.
   - Functionalities include evaluating model performance, analyzing model characteristics, and fine-tuning pre-trained models.

## Notebooks:

1. **distillation_experiments:**
   - Implementation of knowledge distillation, a technique where a smaller student model is trained to mimic the behavior of a larger teacher model, enhancing the student's performance.

2. **mixer_final.ipynb:**
   - Involves loading, evaluating, and modifying Mixer models on the ImageNet dataset.
   - Demonstrates the creation of a collapsible version of a Mixer model, loading different states, modifying the collapsible model's architecture, and evaluating performance.

3. **vgg_final.ipynb:**
   - Involves loading pretrained VGG models (VGG11, VGG13, VGG16, VGG19) from torchvision.
   - Creates collapsible versions of these models, loads different states, and evaluates the performance of both the original and collapsible models on the ImageNet dataset.

4. **vit_final.ipynb:**
   - Conducts sensitivity analysis on a ViT Tiny model by gradually collapsing different fractions of its layers.
   - Evaluates the performance of each step and the final collapsed model on the ImageNet dataset.
