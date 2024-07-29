## README
## The name of the experiment, project and wandb tag are set in the vars section as CLI overrides. 
## These parameters organize how to label the wandb upload and where the model checkpoints are saved. 

export WANDB_ENTITY=aim-harvard
ct_fm_path="/mnt/data1/CT_FM/latest_fm_checkpoints/ct_fm_simclr_segresnetds_22_jul_2024.ckpt"

# ######################### Full - V2 ############################
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/baseline.yaml --trainer#callbacks#0#until_epoch=0 --vars#group="v2"
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks#0#until_epoch=0 --vars#group="v2"
lighter fit --config=./evaluation/totalseg.yaml --trainer#callbacks#0#until_epoch=0 --vars#group="v2" --system#model#trunk#ckpt_path=$ct_fm_path
