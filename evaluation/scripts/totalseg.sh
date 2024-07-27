export WANDB_ENTITY=aim-harvard
ct_fm_path="/mnt/data1/CT_FM/latest_fm_checkpoints/ct_fm_simclr_segresnetds_22_jul_2024.ckpt"


# # ######################### Decoder-only ############################
# lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#strategy=ddp_find_unused_parameters_true --vars#name="suprem_unet" --vars#project="totalseg" --vars#wandb_tag='$["representations"]'
# lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#strategy=ddp_find_unused_parameters_true --vars#name="suprem_segresnet" --vars#project="totalseg" --vars#wandb_tag='$["representations"]'
# lighter fit --config=./evaluation/totalseg.yaml --trainer#strategy=ddp_find_unused_parameters_true --vars#name="ct_fm_segresnet" --vars#project="totalseg" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_tag='$["representations"]'


# # ######################### Few-shot train samples ############################
# # 20-shot 
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/segresnetds.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=2 --vars#name="baseline" --vars#project="totalseg" --vars#wandb_tag='$["few-shot", "20"]' 
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=2 --vars#name="suprem_unet" --vars#project="totalseg" --vars#wandb_tag='$["few-shot", "20"]'
# lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=2
lighter fit --config=./evaluation/totalseg.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=2 --vars#name="ct_fm" --vars#project="totalseg" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_tag='$["few-shot", "20"]'


# # 10-shot
# lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/random_init.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=1 --vars#name="baseline" --vars#project="totalseg" --vars#wandb_tag='$["few-shot", "10"]'
# lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=1 --vars#name="suprem_unet" --vars#project="totalseg" --vars#wandb_tag='$["few-shot", "10"]'
# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None --vars#percentage=1 
# lighter fit --config=./evaluation/totalseg.yaml,./evaluation/frameworks/ct_fm.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=1 --vars#name="ct_fm" --vars#project="totalseg" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_tag='$["few-shot", "10"]'

# # 5-shot
# lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/random_init.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=0.5 --vars#name="baseline" --vars#project="totalseg" --vars#wandb_tag='$["few-shot", "5"]'
# lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=0.5 --vars#name="suprem_unet" --vars#project="totalseg" --vars#wandb_tag='$["few-shot", "5"]'
# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=0.5
# lighter fit --config=./evaluation/totalseg.yaml,./evaluation/frameworks/ct_fm.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=0.5 --vars#name="ct_fm" --vars#project="totalseg" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_tag='$["few-shot", "5"]'



# # ######################### Few-shot annotation groups ############################
# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/random_init.yaml --trainer#callbacks=None --vars#group="cardiac"
# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None --vars#group="cardiac"
# # # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None --vars#group="cardiac"
# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/frameworks/ct_fm.yaml --trainer#callbacks=None --vars#group="cardiac"


# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None --vars#group="organs"
# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/random_init.yaml --trainer#callbacks=None --vars#group="organs"
# # # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None --vars#group="organs"
# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/frameworks/ct_fm.yaml --trainer#callbacks=None --vars#group="organs"

# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/random_init.yaml --trainer#callbacks=None --vars#group="muscles"
# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None --vars#group="muscles"
# # # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None --vars#group="muscles"
# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/frameworks/ct_fm.yaml --trainer#callbacks=None --vars#group="muscles"

# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/random_init.yaml --trainer#callbacks=None --vars#group="vertebra"
# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None --vars#group="vertebra"
# # # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None --vars#group="vertebra"
# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/frameworks/ct_fm.yaml --trainer#callbacks=None --vars#group="vertebra"

# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/random_init.yaml --trainer#callbacks=None --vars#group="muscles"
# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None --vars#group="muscles"
# # # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None --vars#group="muscles"
# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/frameworks/ct_fm.yaml --trainer#callbacks=None --vars#group="muscles"


# ######################### Full ############################
# lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/random_init.yaml --trainer#callbacks#0#until_epoch=0
# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#callbacks#0#until_epoch=0
# # lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks#0#until_epoch=0
# lighter fit --config=./evaluation/totalseg.yaml,./evaluation/frameworks/ct_fm.yaml --trainer#callbacks#0#until_epoch=0
