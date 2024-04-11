CUDA_VISIBLE_DEVICES=0 lighter fit --config_file=./experiments/evaluation/class_base.yaml,./experiments/evaluation/ablations/conrecon.yaml,./experiments/evaluation/ablations/class_overrides.yaml &
CUDA_VISIBLE_DEVICES=1 lighter fit --config_file=./experiments/evaluation/class_base.yaml,./experiments/evaluation/ablations/reconstruction.yaml,./experiments/evaluation/ablations/class_overrides.yaml &
CUDA_VISIBLE_DEVICES=2 lighter fit --config_file=./experiments/evaluation/class_base.yaml,./experiments/evaluation/ablations/vicregl.yaml,./experiments/evaluation/ablations/class_overrides.yaml &
CUDA_VISIBLE_DEVICES=3 lighter fit --config_file=./experiments/evaluation/class_base.yaml,./experiments/evaluation/ablations/simclr.yaml,./experiments/evaluation/ablations/class_overrides.yaml &
wait
