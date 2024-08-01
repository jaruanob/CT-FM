## README
## The name of the experiment, project and wandb tag are set in the vars section as CLI overrides. 
## These parameters organize how to label the wandb upload and where the model checkpoints are saved. 

export WANDB_ENTITY=aim-harvard
ct_fm_path="/mnt/data1/CT_FM/latest_fm_checkpoints/ct_fm_simclr_segresnetds_22_jul_2024.ckpt"


# ######################### Groups - Merlin V2 ############################
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_merlin.yaml,./evaluation/baselines/segresnetds.yaml --vars#name="baseline" --vars#project="totalseg" --vars#wandb_tag='$["merlin", "v2"]'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_merlin.yaml --system#model#trunk#ckpt_path=$ct_fm_path --vars#name="ct_fm" --vars#project="totalseg" --vars#wandb_tag='$["merlin", "v2"]'
