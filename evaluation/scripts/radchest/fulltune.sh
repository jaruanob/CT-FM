# lighter fit --config=./evaluation/radchest.yaml,./evaluation/baselines/random_init_encoder.yaml --system#optimizer#weight_decay=0.0001
# lighter fit --config=./evaluation/radchest.yaml,./evaluation/baselines/random_init_encoder.yaml --system#batch_size=8 --vars#size=[128,224,224]
# lighter fit --config=./evaluation/radchest.yaml,./evaluation/baselines/random_init_encoder.yaml --system#batch_size=72 --vars#size=[64,96,96] 
lighter fit --config=./evaluation/radchest.yaml,./evaluation/baselines/random_init_encoder.yaml --system#scheduler=None
lighter fit --config=./evaluation/radchest.yaml,./evaluation/baselines/random_init_encoder.yaml --vars#init_lr=0.00001
lighter fit --config=./evaluation/radchest.yaml,./evaluation/baselines/random_init_encoder.yaml --vars#init_lr=0.0001
lighter fit --config=./evaluation/radchest.yaml,./evaluation/baselines/random_init_encoder.yaml --vars#init_lr=0.00005

# lighter fit --config=./evaluation/radchest.yaml,./evaluation/frameworks/conrecon.yaml
# lighter fit --config=./evaluation/radchest.yaml,./evaluation/frameworks/simclr.yaml

# lighter fit --config=./evaluation/radchest.yaml,./evaluation/baselines/suprem_unet_encoder.yaml --system#batch_size=24
# lighter fit --config=./evaluation/radchest.yaml,./evaluation/baselines/suprem_segresnet_encoder.yaml --system#batch_size=64
