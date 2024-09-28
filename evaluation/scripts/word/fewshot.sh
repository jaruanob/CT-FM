## README
## The name of the experiment, project and wandb tag are set in the vars section as CLI overrides. 
## These parameters organize how to label the wandb upload and where the model checkpoints are saved. 

export WANDB_ENTITY=aim-harvard
ct_fm_path="/mnt/data1/CT_FM/latest_fm_checkpoints/original/epoch=449-step=225000-v1.ckpt"
# ######################### Few-shot train samples ############################
mode=$1

if [ "$mode" == "train" ]; then
  # lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/suprem_unet.yaml --vars#name="suprem_unet" --vars#project="word" --vars#wandb_group='fewshot_5' --vars#samples=5 --trainer#callbacks#0#until_epoch=0
  lighter fit --config=./evaluation/word.yaml --vars#name="ct_fm" --vars#project="word" --system#model#trunk#ckpt_path=$ct_fm_path --system#model#trunk#model#dsdepth=1 --vars#wandb_group='fewshot_5' --vars#samples=5 --trainer#callbacks#0#until_epoch=0
  lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/segresnet_vista3d.yaml --vars#name="vista3d" --vars#project="word" --vars#wandb_group='fewshot_5' --vars#samples=5 --trainer#callbacks#0#until_epoch=0

  # lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/suprem_unet.yaml --vars#name="suprem_unet" --vars#project="word" --vars#wandb_group='fewshot_10' --vars#samples=10 --trainer#callbacks#0#until_epoch=0
  lighter fit --config=./evaluation/word.yaml --vars#name="ct_fm" --vars#project="word" --system#model#trunk#ckpt_path=$ct_fm_path --system#model#trunk#model#dsdepth=1 --vars#wandb_group='fewshot_10' --vars#samples=10 --trainer#callbacks#0#until_epoch=0
  lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/segresnet_vista3d.yaml --vars#name="vista3d" --vars#project="word" --vars#wandb_group='fewshot_10' --vars#samples=10 --trainer#callbacks#0#until_epoch=0

  # lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/suprem_unet.yaml --vars#name="suprem_unet" --vars#project="word" --vars#wandb_group='fewshot_20' --vars#samples=20 --trainer#callbacks#0#until_epoch=0
  lighter fit --config=./evaluation/word.yaml --vars#name="ct_fm" --vars#project="word" --system#model#trunk#ckpt_path=$ct_fm_path --system#model#trunk#model#dsdepth=1 --vars#wandb_group='fewshot_20' --vars#samples=20 --trainer#callbacks#0#until_epoch=0
  lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/segresnet_vista3d.yaml --vars#name="vista3d" --vars#project="word" --vars#wandb_group='fewshot_20' --vars#samples=20 --trainer#callbacks#0#until_epoch=0

  # lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/suprem_unet.yaml --vars#name="suprem_unet" --vars#project="word" --vars#wandb_group='fewshot_50' --vars#samples=50 --trainer#callbacks#0#until_epoch=0
  lighter fit --config=./evaluation/word.yaml --vars#name="ct_fm" --vars#project="word" --system#model#trunk#ckpt_path=$ct_fm_path --system#model#trunk#model#dsdepth=1 --vars#wandb_group='fewshot_50' --vars#samples=50 --trainer#callbacks#0#until_epoch=0
  lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/segresnet_vista3d.yaml --vars#name="vista3d" --vars#project="word" --vars#wandb_group='fewshot_50' --vars#samples=50 --trainer#callbacks#0#until_epoch=0


elif [ "$mode" == "predict" ]; then
  lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/suprem_unet.yaml --vars#name="suprem_unet" --vars#project="word" --vars#wandb_group='fewshot_5' --vars#samples=5
  lighter fit --config=./evaluation/word.yaml --vars#name="ct_fm" --vars#project="word" --system#model#trunk#ckpt_path=$ct_fm_path --system#model#trunk#model#dsdepth=1 --vars#wandb_group='fewshot_5' --vars#samples=5
  lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/segresnet_vista3d.yaml --vars#name="vista3d" --vars#project="word" --vars#wandb_group='fewshot_5' --vars#samples=5

  lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/suprem_unet.yaml --vars#name="suprem_unet" --vars#project="word" --vars#wandb_group='fewshot_10' --vars#samples=10
  lighter fit --config=./evaluation/word.yaml --vars#name="ct_fm" --vars#project="word" --system#model#trunk#ckpt_path=$ct_fm_path --system#model#trunk#model#dsdepth=1 --vars#wandb_group='fewshot_10' --vars#samples=10
  lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/segresnet_vista3d.yaml --vars#name="vista3d" --vars#project="word" --vars#wandb_group='fewshot_10' --vars#samples=10

  lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/suprem_unet.yaml --vars#name="suprem_unet" --vars#project="word" --vars#wandb_group='fewshot_20' --vars#samples=20
  lighter fit --config=./evaluation/word.yaml --vars#name="ct_fm" --vars#project="word" --system#model#trunk#ckpt_path=$ct_fm_path --system#model#trunk#model#dsdepth=1 --vars#wandb_group='fewshot_20' --vars#samples=20
  lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/segresnet_vista3d.yaml --vars#name="vista3d" --vars#project="word" --vars#wandb_group='fewshot_20' --vars#samples=20

  lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/suprem_unet.yaml --vars#name="suprem_unet" --vars#project="word" --vars#wandb_group='fewshot_50' --vars#samples=50
  lighter fit --config=./evaluation/word.yaml --vars#name="ct_fm" --vars#project="word" --system#model#trunk#ckpt_path=$ct_fm_path --system#model#trunk#model#dsdepth=1 --vars#wandb_group='fewshot_50' --vars#samples=50
  lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/segresnet_vista3d.yaml --vars#name="vista3d" --vars#project="word" --vars#wandb_group='fewshot_50' --vars#samples=50

else
  echo "Invalid mode. Please use 'train' or 'predict'."
fi
