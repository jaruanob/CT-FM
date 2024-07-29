## README
## The name of the experiment, project and wandb tag are set in the vars section as CLI overrides. 
## These parameters organize how to label the wandb upload and where the model checkpoints are saved. 

export WANDB_ENTITY=aim-harvard
ct_fm_path="/mnt/data1/CT_FM/latest_fm_checkpoints/ct_fm_simclr_segresnetds_22_jul_2024.ckpt"

# ######################### Decoder-only ############################
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#strategy=ddp_find_unused_parameters_true --vars#name="suprem_unet" --vars#project="totalseg" --vars#wandb_tag='$["representations"]'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#strategy=ddp_find_unused_parameters_true --vars#name="suprem_segresnet" --vars#project="totalseg" --vars#wandb_tag='$["representations"]'
lighter fit --config=./evaluation/totalseg.yaml --trainer#strategy=ddp_find_unused_parameters_true --vars#name="ct_fm_segresnet" --vars#project="totalseg" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_tag='$["representations"]'
