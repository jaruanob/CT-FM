# Pre-training CT-FM

Before you begin, ensure you have downloaded your data as explained in the [Data Instructions](./data.md). It is also a good idea to review the [lighter documentation](https://github.com/project-lighter/lighter) since our training configurations are based on its guidelines.

## Pre-training Experiment Configurations

All configuration files for pre-training are located in the experiments directory. In particular, check out two key folders:

- [ablations](https://github.com/project-lighter/CT-FM/tree/main/experiments/ablations) folder, which contains setups for testing various experimental approaches.
- [fm](https://github.com/project-lighter/CT-FM/tree/main/experiments/fm) folder, which holds the finalized configurations for the pre-training run.


## Final Pre-training Setup

### Running the Experiment

After all adjustments have been made, navigate to the root directory of the CT-FM project and execute the following command to begin pre-training:


```bash title="Pretraining command"
lighter fit --config=./experiments/fm/base.yaml, #(1)!
./experiments/fm/frameworks/intrasample_simclr.yaml, #(2)!
./experiments/fm/backbones/segresenc.yaml #(3)!
```

1.    The [file](https://github.com/project-lighter/CT-FM/tree/main/experiments/fm/base.yaml) establishes the core settings for pre-training the CT-FM model by defining:
      - **Variables:** Core parameters such as voxel spacing.
      - **Trainer Settings:** Parameters including 500 epochs, batch limits, GPU configuration, mixed precision, logging via WandB, and checkpoint callbacks.
      - **System Settings:** The model placeholder, optimizer (AdamW), learning rate scheduler (WarmupCosineSchedule), and dataloader setup for safely handling your dataset.
      - **Adapters:** Methods for batch processing and loss computation.
      In essence, `base.yaml` serves as the foundation upon which the entire pre-training process is built.

2.    The [file](https://github.com/project-lighter/CT-FM/tree/main/experiments/fm/frameworks/intrasample_simclr.yaml) configures the self-supervised SimCLR framework used during pre-training. It includes:
      - **Model & Criterion:** Defines the CT-FM model and applies a contrastive loss function with a specified temperature.
      - **Data Augmentation Pipeline:** Implements a series of transformations (such as random crops, flips, and intensity adjustments) to generate multiple augmented views from each input image.
      This configuration augments the base setup with specialized self-supervised learning components.

3.    The [file](https://github.com/project-lighter/CT-FM/tree/main/experiments/fm/backbones/segresenc.yaml) sets up the backbone for the CT-FM model. It includes:
      - **Backbone Identification:** Sets the variable `BACKBONE_NAME` to `"SegResNetDS"`.
      - **Architectural Details:** Configures the SegResNet encoder (via `monai.networks.nets.segresnet_ds.SegResEncoder`) by specifying parameters like spatial dimensions, input channels, initial filters, and block structures.
      - **Integration with Base Config:** Uses shared variable mappings (such as `NUM_FTRS_BY_BACKBONE`) and logger identifiers from `base.yaml` to ensure smooth integration.
      This configuration provides the essential backbone architecture for complete model training.

!!! note
      Click on the :material-star-four-points-circle: symbols to learn more about each yaml file

## Customization Before Training

Before running the experiment, make sure to update a few paths in the `base.yaml` file:
- Replace the paths for `save_dir` and `dirpath` with your chosen directories for saving logs and checkpoints.
- Update the path to the `scan_list.pkl` file with the one generated during the data preparation phase.

Additionally, adjust the training parameters under the `trainer:` key (such as the number of GPUs, batch size, and training duration) to suit your system’s resources and your experimental needs.


This command starts the pre-training process using your customized configurations.