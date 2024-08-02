## README
## The name of the experiment, project and wandb tag are set in the vars section as CLI overrides. 
## These parameters organize how to label the wandb upload and where the model checkpoints are saved. 

export WANDB_ENTITY=aim-harvard
ct_fm_path="/mnt/data1/CT_FM/latest_fm_checkpoints/ct_fm_simclr_segresnetds_22_jul_2024.ckpt"

# ######################### Groups - V1 (SAT split) ############################
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_sat.yaml,./evaluation/baselines/segresnetds.yaml --vars#group="cardiac_v1" --trainer#callbacks#0#until_epoch=0 --vars#name="baseline" --vars#project="totalseg" --vars#wandb_group='cardiac_sat_v1'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_sat.yaml,./evaluation/baselines/suprem_unet.yaml --vars#group="cardiac_v1" --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unet" --vars#project="totalseg" --vars#wandb_group='cardiac_sat_v1'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_sat.yaml --trainer#callbacks#0#until_epoch=0 --vars#group="cardiac_v1" --vars#name="ct_fm" --vars#project="totalseg" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_group='cardiac_sat_v1'

lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_sat.yaml,./evaluation/baselines/segresnetds.yaml --vars#group="organs_v1" --trainer#callbacks#0#until_epoch=0 --vars#name="baseline" --vars#project="totalseg" --vars#wandb_group='organs_sat_v1'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_sat.yaml,./evaluation/baselines/suprem_unet.yaml --vars#group="organs_v1" --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unet" --vars#project="totalseg" --vars#wandb_group='organs_sat_v1'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_sat.yaml --trainer#callbacks#0#until_epoch=0 --vars#group="organs_v1" --vars#name="ct_fm" --vars#project="totalseg" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_group='organs_sat_v1'

lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_sat.yaml,./evaluation/baselines/segresnetds.yaml --vars#group="muscles_v1" --trainer#callbacks#0#until_epoch=0 --vars#name="baseline" --vars#project="totalseg" --vars#wandb_group='muscles_sat_v1'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_sat.yaml,./evaluation/baselines/suprem_unet.yaml --vars#group="muscles_v1" --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unet" --vars#project="totalseg" --vars#wandb_group='muscles_sat_v1'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_sat.yaml --trainer#callbacks#0#until_epoch=0 --vars#group="muscles_v1" --vars#name="ct_fm" --vars#project="totalseg" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_group='muscles_sat_v1'

lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_sat.yaml,./evaluation/baselines/segresnetds.yaml --vars#group="vertebra_v1" --trainer#callbacks#0#until_epoch=0 --vars#name="baseline" --vars#project="totalseg" --vars#wandb_group='vertebra_sat_v1'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_sat.yaml,./evaluation/baselines/suprem_unet.yaml --vars#group="vertebra_v1" --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unet" --vars#project="totalseg" --vars#wandb_group='vertebra_sat_v1'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_sat.yaml --trainer#callbacks#0#until_epoch=0 --vars#group="vertebra_v1" --vars#name="ct_fm" --vars#project="totalseg" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_group='vertebra_sat_v1'

lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_sat.yaml,./evaluation/baselines/segresnetds.yaml --vars#group="ribs_v1" --trainer#callbacks#0#until_epoch=0 --vars#name="baseline" --vars#project="totalseg" --vars#wandb_group='ribs_sat_v1'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_sat.yaml,./evaluation/baselines/suprem_unet.yaml --vars#group="ribs_v1" --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unet" --vars#project="totalseg" --vars#wandb_group='ribs_sat_v1'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_sat.yaml --trainer#callbacks#0#until_epoch=0 --vars#group="ribs_v1" --vars#name="ct_fm" --vars#project="totalseg" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_group='ribs_sat_v1'
