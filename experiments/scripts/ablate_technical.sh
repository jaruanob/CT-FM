export CUDA_VISIBLE_DEVICES=0,1,3
# lighter fit --config=./experiments/ablations/base.yaml,./experiments/ablations/frameworks/conrecon.yaml,./experiments/ablations/backbones/segresnetds_w_embedding.yaml; pkill -9 -f lighter
# lighter fit --config=./experiments/ablations/base.yaml,./experiments/ablations/frameworks/simclr_cross_sample.yaml,./experiments/ablations/backbones/segresenc.yaml;pkill -9 -f lighter
# lighter fit --config=./experiments/ablations/base.yaml,./experiments/ablations/frameworks/simclr.yaml,./experiments/ablations/backbones/segresenc.yaml;pkill -9 -f lighter
# lighter fit --config=./experiments/ablations/base.yaml,./experiments/ablations/frameworks/reconstruction.yaml,./experiments/ablations/backbones/segresnetds.yaml; pkill -9 -f lighter
lighter fit --config=./experiments/ablations/base.yaml,./experiments/ablations/frameworks/vicreg_intrasample.yaml,./experiments/ablations/backbones/segresenc.yaml
lighter fit --config=./experiments/ablations/base.yaml,./experiments/ablations/frameworks/simsiam.yaml,./experiments/ablations/backbones/segresenc.yaml


# lighter fit --config=./experiments/ablations/base.yaml,./experiments/ablations/frameworks/simclr_cross_sample_small_patch.yaml,./experiments/ablations/backbones/segresenc.yaml; pkill -9 -f lighter
# lighter fit --config=./experiments/ablations/base.yaml,./experiments/ablations/frameworks/simclr_cross_sample.yaml,./experiments/ablations/backbones/segresenc.yaml;pkill -9 -f lighter
# lighter fit --config=./experiments/ablations/base.yaml,./experiments/ablations/frameworks/reconstruction_small_patch.yaml,./experiments/ablations/backbones/segresnetds.yaml; pkill -9 -f lighter


# lighter fit --config=./experiments/ablations/base.yaml,./experiments/ablations/frameworks/simsiam_intrasample.yaml,./experiments/ablations/backbones/segresenc.yaml; pkill -9 -f lighter
