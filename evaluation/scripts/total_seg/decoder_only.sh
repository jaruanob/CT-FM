export WANDB_ENTITY=aim-harvard
# Lightweight decoder with upsampling layers non-trainable (halves the number of params)
lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/conrecon.yaml --system#model#trunk#model#upsample_mode="nontrainable"
lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/simclr.yaml --system#model#trunk#model#upsample_mode="nontrainable"
lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/reconstruction.yaml --system#model#trunk#model#upsample_mode="nontrainable"


lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/baselines/suprem_unet.yaml
lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/baselines/suprem_segresnet.yaml
lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/conrecon.yaml
lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/simclr.yaml
lighter fit --config_file=./experiments/evaluation/totalseg.yaml,./experiments/evaluation/frameworks/reconstruction.yaml


