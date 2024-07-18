# 
CUDA_VISIBLE_DEVICES=0 lighter fit --config=./evaluation/lits.yaml,./evaluation/baselines/random_init.yaml 
CUDA_VISIBLE_DEVICES=1 lighter fit --config=./evaluation/lits.yaml,./evaluation/baselines/random_init.yaml 
CUDA_VISIBLE_DEVICES=2 lighter fit --config=./evaluation/lits.yaml,./evaluation/baselines/random_init.yaml 
CUDA_VISIBLE_DEVICES=3 lighter fit --config=./evaluation/lits.yaml,./evaluation/baselines/random_init.yaml 