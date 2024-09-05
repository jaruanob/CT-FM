export WANDB_ENTITY="aim-harvard"
lighter fit --config=./evaluation/lits.yaml,./evaluation/baselines/random_init.yaml
pkill -9 -f lighter

lighter fit --config=./evaluation/lits.yaml,./evaluation/baselines/unet_suprem.yaml --trainer#callbacks=None
pkill -9 -f lighter

lighter fit --config=./evaluation/lits.yaml,./evaluation/frameworks/conrecon.yaml
pkill -9 -f lighter

lighter fit --config=./evaluation/lits.yaml,./evaluation/frameworks/simclr.yaml
pkill -9 -f lighter

lighter fit --config=./evaluation/lits.yaml,./evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None
pkill -9 -f lighter

lighter fit --config=./evaluation/lits.yaml,./evaluation/frameworks/reconstruction.yaml