[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/project-lighter)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2501.09001)

# CT-FM: A 3D Image-Based Foundation Model for Radiological Tasks

![CT-FM Overview](./assets/Figure1.png)

## Introduction
This repository contains the code and resources for CT-FM, a 3D image-based pre-trained foundation model designed for various radiological tasks. CT-FM is trained using self-supervised learning (SSL) on a large dataset of 148,000 CT scans. This model aims to address a range of tasks, including whole-body segmentation, tumor segmentation, head CT triage, and medical image retrieval. This work builds upon previous efforts in radiological AI, shifting from task-specific expert models to unified foundation models for broader adaptability and efficiency.

## Key Innovations

*   **Large-Scale 3D Pretraining:** Emphasis on 3D data rather than traditional 2D datasets.
*   **Task-Agnostic Training:** Enabling transferability across various radiological tasks.
*   **Open Source:** Model weights, data, and code are shared for collaborative development.

## Training
CT-FM largely relies on the [lighter](https://github.com/project-lighter/lighter) package. Detailed descriptions on running the pretraining and downstream task specific finetuning can be found in the [replication guide](./replication-guide)