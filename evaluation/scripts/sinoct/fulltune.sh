## README
## The name of the experiment, project and wandb tag are set in the vars section as CLI overrides. 
## These parameters organize how to label the wandb upload and where the model checkpoints are saved. 

export WANDB_ENTITY=aim-harvard
wandb_group=$(date +'%Y-%m-%d_%H-%M-%S')

# Weights paths
ct_fm_path="/mnt/data16/ibro/IDC_SSL_CT/runs/fm/checkpoints/CT_FM_SimCLR_SegResNetDS/epoch=449-step=225000-v1.ckpt"
suprem_path="/home/Projects/CT_FM/baselines/SuPreM_SegResNet/supervised_suprem_segresnet_2100.pth"
vista3d_path="/home/Projects/CT_FM/baselines/VISTA3D/vista3d/models/model.pt"

lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_random.yaml --trainer#callbacks#0#until_epoch=0 --vars#name="baseline_100pct" --vars#project="sinoct" --vars#wandb_group=$wandb_group
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm_100pct" --vars#project="sinoct" --vars#wandb_group=$wandb_group --system#model#trunk#ckpt_path=$ct_fm_path
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_suprem.yaml --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_segresencoder_100pct" --vars#project="sinoct" --vars#wandb_group=$wandb_group -system#model#trunk#ckpt_path=$suprem_path
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --trainer#callbacks#0#until_epoch=0 --vars#name="vista3d_segresencoder_100pct" --vars#project="sinoct" --vars#wandb_group=$wandb_group -system#model#trunk#ckpt_path=$vista3d_path

lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=5 --trainer#callbacks#0#until_epoch=0 --vars#name="baseline_5pct" --vars#project="sinoct" --vars#wandb_group=$wandb_group
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=5 --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm_5pct" --vars#project="sinoct" --vars#wandb_group=$wandb_group --system#model#trunk#ckpt_path=$ct_fm_path
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_suprem.yaml --vars#percentage=5 --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_segresencoder_5pct" --vars#project="sinoct" --vars#wandb_group=$wandb_group -system#model#trunk#ckpt_path=$suprem_path
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#percentage=5 --trainer#callbacks#0#until_epoch=0 --vars#name="vista3d_segresencoder_5pct" --vars#project="sinoct" --vars#wandb_group=$wandb_group -system#model#trunk#ckpt_path=$vista3d_path

lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=2 --trainer#callbacks#0#until_epoch=0 --vars#name="baseline_2pct" --vars#project="sinoct" --vars#wandb_group=$wandb_group
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=2 --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm_2pct" --vars#project="sinoct" --vars#wandb_group=$wandb_group --system#model#trunk#ckpt_path=$ct_fm_path
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_suprem.yaml --vars#percentage=2 --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_segresencoder_2pct" --vars#project="sinoct" --vars#wandb_group=$wandb_group -system#model#trunk#ckpt_path=$suprem_path
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#percentage=2 --trainer#callbacks#0#until_epoch=0 --vars#name="vista3d_segresencoder_2pct" --vars#project="sinoct" --vars#wandb_group=$wandb_group -system#model#trunk#ckpt_path=$vista3d_path

lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=1 --trainer#callbacks#0#until_epoch=0 --vars#name="baseline_1pct" --vars#project="sinoct" --vars#wandb_group=$wandb_group
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=1 --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm_1pct" --vars#project="sinoct" --vars#wandb_group=$wandb_group --system#model#trunk#ckpt_path=$ct_fm_path
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_suprem.yaml --vars#percentage=1 --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_segresencoder_1pct" --vars#project="sinoct" --vars#wandb_group=$wandb_group -system#model#trunk#ckpt_path=$suprem_path
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#percentage=1 --trainer#callbacks#0#until_epoch=0 --vars#name="vista3d_segresencoder_1pct" --vars#project="sinoct" --vars#wandb_group=$wandb_group -system#model#trunk#ckpt_path=$vista3d_path
