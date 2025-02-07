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

### Enabling Prediction Mode

To switch from training to prediction, replace the `fit` command with the `predict` command and append the prediction override configuration file.

<span class="md-tooltip" title="Append <code>,./evaluation/overrides/totalseg_predict_overrides.yaml</code> to the config list"><strong>Added:</strong></span> Append <code>,./evaluation/overrides/totalseg_predict_overrides.yaml</code> to the config list.

<span class="md-tooltip" title="Removed the <code>--trainer#callbacks#0#until_epoch=0</code> flag since the new callback handles prediction mode"><strong>Removed:</strong></span> the <code>--trainer#callbacks#0#until_epoch=0</code> flag.

**Example Transformation:**

Original command:
```
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/segresnetds_ctfm.yaml --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm" --vars#project="totalseg" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_group='vista_v2'
```

Modified prediction command:
```
lighter predict --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/segresnetds_ctfm.yaml,./evaluation/overrides/totalseg_predict_overrides.yaml --vars#name="ct_fm" --vars#project="totalseg" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_group='vista_v2'
```

To override the model checkpoint directory during prediction, add:
```
--args#predict#ckpt_path=<path>
```

## Tumor Segmentation - Auto3DSeg
