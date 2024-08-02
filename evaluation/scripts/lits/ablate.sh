lighter fit --config=./evaluation/lits.yaml,./evaluation/baselines/random_init.yaml --vars#patch_size=[128,160,160] --system#batch_size=8
lighter fit --config=./evaluation/lits.yaml,./evaluation/baselines/random_init.yaml --vars#patch_size=[96,128,128] --system#batch_size=16
lighter fit --config=./evaluation/lits.yaml,./evaluation/baselines/random_init.yaml --vars#patch_size=[128,160,160] --system#batch_size=8 --vars#ratio=[0.4, 0.1, 0.5]
