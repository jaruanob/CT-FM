export WANDB_ENTITY=aim-harvard
# Lightweight decoder with upsampling layers non-trainable (halves the number of params)
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/frameworks/conrecon.yaml --system#model#trunk#model#upsample_mode="nontrainable"
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/frameworks/simclr.yaml --system#model#trunk#model#upsample_mode="nontrainable"
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/frameworks/reconstruction.yaml --system#model#trunk#model#upsample_mode="nontrainable"


lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_unet.yaml
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/baselines/suprem_segresnet.yaml
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/frameworks/conrecon.yaml
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/frameworks/simclr.yaml
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/frameworks/reconstruction.yaml


