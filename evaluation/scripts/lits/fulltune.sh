export WANDB_ENTITY="aim-harvard"
lighter fit --config_file=./experiments/evaluation/lits.yaml,./experiments/evaluation/baselines/random_init.yaml
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/lits.yaml,./experiments/evaluation/baselines/suprem_unet.yaml --trainer#callbacks=None
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/lits.yaml,./experiments/evaluation/frameworks/conrecon.yaml
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/lits.yaml,./experiments/evaluation/frameworks/simclr.yaml
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/lits.yaml,./experiments/evaluation/baselines/suprem_segresnet.yaml --trainer#callbacks=None
pkill -9 -f lighter

lighter fit --config_file=./experiments/evaluation/lits.yaml,./experiments/evaluation/frameworks/reconstruction.yaml