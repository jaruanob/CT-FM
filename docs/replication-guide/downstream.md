# Downstream Task Adaptation

Our pre-trained CT-FM model has been adapted to three fine-tuned downstream tasks as well as several additional zero‐shot tasks. While most downstream experiments leverage the Lighter framework, tumor segmentation is handled using Auto3DSeg.

## Whole Body Segmentation

In line with the configuration-based approach detailed in [Pretraining](./pretraining.md), we provide YAML config files for downstream adaptation. To facilitate thorough comparisons, a suite of shell scripts with the relevant configuration components is available. These can be found in the [evaluation](https://github.com/project-lighter/CT-FM/tree/main/evaluation) directory under “scripts.”

[View All Scripts](https://github.com/project-lighter/CT-FM/tree/main/evaluation/scripts){.md-button}


For TotalSeg experiments, refer to the scripts in the totalseg folder:
<div class="grid cards" markdown>
- **Full Finetuning on TotalSegmentatorV2:**  
  [fulltune.sh](https://github.com/project-lighter/CT-FM/tree/main/evaluation/scripts/totalseg/fulltune.sh)

- **Finetuning on the Merlin Split:**  
  [merlin.sh](https://github.com/project-lighter/CT-FM/tree/main/evaluation/scripts/totalseg/merlin.sh)

- **Few-Shot Fine-Tuning:**  
  [fewshot.sh](https://github.com/project-lighter/CT-FM/tree/main/evaluation/scripts/totalseg/fewshot.sh)

- **Pre-Training Checkpoint Selection:**  
  [checkpoint_selection.sh](https://github.com/project-lighter/CT-FM/tree/main/evaluation/scripts/totalseg/checkpoint_selection.sh)

- **Pre-Training Ablations:**  
  [pretraining_evaluation.sh](https://github.com/project-lighter/CT-FM/tree/main/evaluation/scripts/totalseg/pretraining_evaluation.sh)
</div>

!!! tip "Enabling Prediction Mode"

    To switch from training to prediction mode:
    - Replace the `fit` command with the `predict` command.
    - Append the prediction override configuration file `./evaluation/overrides/totalseg_predict_overrides.yaml` to your config list.
    - Remove the `--trainer#callbacks#0#until_epoch=0` flag since the new callback now handles prediction mode.

**Example Transformation:**

Original command:
```
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/segresnetds_ctfm.yaml --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm" --vars#project="totalseg" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_group='vista_v2'
```

Modified prediction command:
```
lighter predict --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/segresnetds_ctfm.yaml,./evaluation/overrides/totalseg_predict_overrides.yaml --vars#name="ct_fm" --vars#project="totalseg" --vars#wandb_group='vista_v2'
```

By default the predict command uses the checkpoint location mentioned while running the fit pipeline.
If you have a different checkpoint location, to override the model checkpoint directory during prediction, add:
```
--args#predict#ckpt_path=<path>
```

## Tumor Segmentation with Auto3DSeg

Tumor segmentation is performed using Auto3DSeg—a robust segmentation workflow provided by MONAI. This pipeline is designed to simplify segmentation tasks and can be explored further in the official [MONAI Auto3DSeg Tutorial](https://github.com/Project-MONAI/tutorials/blob/main/auto3dseg/README.md).

### Workflow Overview

Auto3DSeg operates by running an AutoRunner that takes a configuration file (typically named task.yaml) as input. This file contains all the necessary parameters to handle preprocessing, training, and validation stages of your segmentation task.

### Model Details

Our experiments focus on the segresnet_0 model variant, which is set up for single-fold training and validation. We run the baseline model using the default Auto3DSeg configuration. However, when integrating our CT-FM model into the pipeline, we make the following two key modifications:

- **Orientation Adjustment:**  
  We change the default image orientation by setting the axcodes to `SPL`.
  
- **Checkpoint Specification:**  
  The path to the pre-trained model checkpoint is provided via the `ckpt_path` field in the hyper_parameters.yaml file.

These adjustments allow us to directly benchmark the effectiveness of the pre-trained CT-FM model within the Auto3DSeg pipeline without necessitating major changes to the existing workflow.

!!! tip "Customizing Your Pipeline"
    By simply modifying the orientation and specifying the checkpoint path, you can leverage the power of pre-trained models in the Auto3DSeg setup. This makes it easy to compare different configurations and accelerate your experimentation process.

