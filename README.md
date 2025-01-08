# CT-FM: A 3D Image-Based Foundation Model for Radiological Tasks

## Introduction

This repository contains the code and resources for CT-FM, a 3D image-based pre-trained foundation model designed for various radiological tasks. CT-FM is trained using self-supervised learning (SSL) on a large dataset of 148,000 CT scans. This model aims to address a range of tasks, including whole-body segmentation, tumor segmentation, head CT triage, and medical image retrieval. This work builds upon previous efforts in radiological AI, shifting from task-specific expert models to unified foundation models for broader adaptability and efficiency.

## Key Innovations

*   **Large-Scale 3D Pretraining:** Emphasis on 3D data rather than traditional 2D datasets.
*   **Task-Agnostic Training:** Enabling transferability across various radiological tasks.
*   **Open Source:** Model weights, data, and code are shared for collaborative development.

## Results

CT-FM has been evaluated on several downstream tasks, demonstrating strong performance:

1.  **Whole-Body Segmentation**
    *   Evaluated on the TotalSegmentator dataset (117 anatomical labels across 1,228 scans).
    *   Achieved a mean Dice coefficient of 0.898, outperforming baseline and SuPREM models on most anatomical regions. Note that nnUnet showed better results due to multi-fold cross-validation and ensemble strategies, which CT-FM didn’t employ.

2.  **Cancer Tumor Segmentation**
    *   Benchmarked on the Medical Segmentation Decathlon dataset for lung, hepatic, and pancreatic tumors.
    *   Demonstrated improved Dice scores and Average Surface Distance (ASD) in segmentation tasks for lung and hepatic tumors. For pancreatic tumors, ASD improvements were noted despite comparable Dice scores to baselines.

3.  **Head CT Triage**
    *   Evaluated on SinoCT (9,000 scans) and CQ500 (491 scans) datasets for normal/abnormal classification.
    *   Achieved an F1 score of 0.776 on SinoCT and 0.754 on CQ500, surpassing random baselines but slightly underperforming the SuPREM model in some metrics.

4.  **Medical Image Retrieval**
    *   Tested on OrganMNIST3D and 3D-MIR datasets.
    *   Outperformed baselines in retrieving scans with similar anatomical regions and lesion characteristics. Achieved top precision scores in lesion-based retrieval tasks.

5.  **Anatomical Clustering and Semantic Search**
    *   Showed inherent clustering of anatomical features in embedding space.
    *   Facilitated fine-grained semantic searches, linking specific anatomical regions across scans.

6.  **Stability and Robustness**
    *   Demonstrated consistent performance across test-retest datasets, showcasing robustness to variations in acquisition parameters.

## Methods

### Pretraining Strategy

CT-FM is pre-trained using a modified SimCLR framework for self-supervised learning:

*   **Intra-sample Contrastive Learning:** Focuses on patches within the same sample to learn spatial semantics.
*   **Augmentation Strategies:** Utilizes augmentations like random cropping, histogram shifting, and intensity scaling.
*   **Pretraining Details:** Pretrained for 500 epochs on 148,000 CT scans, selecting the best checkpoint at epoch 449.

### Fine-tuning for Downstream Tasks

1.  **Segmentation:**
    *   Utilizes the SegResNet architecture with Dice score and cross-entropy loss.
    *   Trained for 300 epochs with augmentations like affine transformations and Gaussian noise.

2.  **Classification:**
    *   Implemented using SegResNet or UNetEncoder as the backbone, optimized with Binary Cross-Entropy loss.
    *   Preprocessing includes windowing levels specific to CT scan features (blood, subdural, stroke, bone).

3.  **Retrieval:**
    *   Embeddings generated for training data were compared using cosine similarity for retrieval tasks.


## Applications for Code Documentation

This section provides guidance on how to use the code for various tasks.

The structure of the repository is as below,
```bash
├── README.md
├── __init__.py
├── callbacks/
├── data/
├── evaluation/
├── experiments/
│   ├── ablations/
│   ├── fm/
│   └── scripts/
├── loss/
├── meta-evaluation/
├── metrics/
├── models/
├── notebooks/
├── semantic-search-app/
├── transforms/
```

The `experiments/` directory contains the following subdirectories:
*   `ablations/`: Contains code for ablation studies.
*   `fm/`: Contains code for the foundation model pre-training.
*   `scripts/`: Contains scripts for running experiments.

The `experiments/ablations/` directory contains configuration files for ablation studies.
    *   `base.yaml`: Defines the base configuration for all ablation experiments, including the trainer, system, optimizer, scheduler, and datasets.
    *   `backbones/`: Contains configuration files for different backbones used in the ablation studies.
        *   `resnet50x2.yaml`: Configuration for a ResNet50x2 backbone.
        *   `segresenc.yaml`: Configuration for a SegResNet encoder backbone.
        *   `segresnetds_w_embedding.yaml`: Configuration for a SegResNetDS backbone with embedding.
        *   `segresnetds.yaml`: Configuration for a SegResNetDS backbone.
    *   `frameworks/`: Contains configuration files for different self-supervised learning frameworks used in the ablation studies.
        *   `conrecon.yaml`: Configuration for the ConRecon framework.
        *   `reconstruction.yaml`: Configuration for a reconstruction-based framework.
        *   `simclr_intrasample.yaml`: Configuration for the SimCLR framework with intra-sample contrastive learning.
        *   `simclr.yaml`: Configuration for the SimCLR framework.
        *   `simsiam_intrasample.yaml`: Configuration for the SimSiam framework with intra-sample contrastive learning.
        *   `simsiam.yaml`: Configuration for the SimSiam framework.
        *   `vicreg_intrasample.yaml`: Configuration for the VICReg framework with intra-sample contrastive learning.
        *   `vicreg.yaml`: Configuration for the VICReg framework.
        *   `vicregl.yaml`: Configuration for the VICRegL framework.

The `experiments/fm/` directory contains configuration files for pre-training the foundation model.
    *   `base.yaml`: Defines the base configuration for pre-training, including the trainer, system, optimizer, scheduler, and datasets.
    *   `backbones/`: Contains configuration files for different backbones used in pre-training.
        *   `segresenc.yaml`: Configuration for a SegResNet encoder backbone.
        *   `segresnetds_w_embedding.yaml`: Configuration for a SegResNetDS backbone with embedding.
    *   `frameworks/`: Contains configuration files for different self-supervised learning frameworks used in pre-training.
        *   `conrecon.yaml`: Configuration for the ConRecon framework.
        *   `simclr.yaml`: Configuration for the SimCLR framework.

The `experiments/scripts/` directory contains scripts for running experiments.
    *   `ablate_technical.sh`: A shell script for running ablation experiments.




This documentation provides a comprehensive overview of the CT-FM project. For more detailed information, please refer to the source code and the original research paper.
