CUDA_VISIBLE_DEVICES=0 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/baselines/suprem_segresnet_encoder.yaml --CONSTANTS#dataset="medmnist.NoduleMNIST3D" --CONSTANTS#num_classes=2 &
CUDA_VISIBLE_DEVICES=1 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/baselines/suprem_unet_encoder.yaml --CONSTANTS#dataset="medmnist.NoduleMNIST3D" --CONSTANTS#num_classes=2 &

CUDA_VISIBLE_DEVICES=2 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/baselines/suprem_segresnet_encoder.yaml --CONSTANTS#dataset="medmnist.FractureMNIST3D" --CONSTANTS#num_classes=3 &
CUDA_VISIBLE_DEVICES=3 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/baselines/suprem_unet_encoder.yaml --CONSTANTS#dataset="medmnist.FractureMNIST3D" --CONSTANTS#num_classes=3 &
wait

CUDA_VISIBLE_DEVICES=0 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/baselines/suprem_segresnet_encoder.yaml --CONSTANTS#dataset="medmnist.OrganMNIST3D" --CONSTANTS#num_classes=11 &
CUDA_VISIBLE_DEVICES=1 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/baselines/suprem_unet_encoder.yaml --CONSTANTS#dataset="medmnist.OrganMNIST3D" --CONSTANTS#num_classes=11 &


# CUDA_VISIBLE_DEVICES=0 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/frameworks/conrecon_L1.yaml --CONSTANTS#dataset="medmnist.NoduleMNIST3D" --CONSTANTS#num_classes=2 &
CUDA_VISIBLE_DEVICES=2 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/frameworks/conrecon.yaml --CONSTANTS#dataset="medmnist.NoduleMNIST3D" --CONSTANTS#num_classes=2 &
CUDA_VISIBLE_DEVICES=3 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/frameworks/reconstruction.yaml --CONSTANTS#dataset="medmnist.NoduleMNIST3D" --CONSTANTS#num_classes=2 &
wait 
CUDA_VISIBLE_DEVICES=0 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/frameworks/simclr.yaml --CONSTANTS#dataset="medmnist.NoduleMNIST3D" --CONSTANTS#num_classes=2 &


# CUDA_VISIBLE_DEVICES=1 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/frameworks/conrecon_L1.yaml --CONSTANTS#dataset="medmnist.FractureMNIST3D" --CONSTANTS#num_classes=3 &
CUDA_VISIBLE_DEVICES=1 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/frameworks/conrecon.yaml --CONSTANTS#dataset="medmnist.FractureMNIST3D" --CONSTANTS#num_classes=3 &
CUDA_VISIBLE_DEVICES=2 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/frameworks/reconstruction.yaml --CONSTANTS#dataset="medmnist.FractureMNIST3D" --CONSTANTS#num_classes=3 &
CUDA_VISIBLE_DEVICES=3 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/frameworks/simclr.yaml --CONSTANTS#dataset="medmnist.FractureMNIST3D" --CONSTANTS#num_classes=3 &

wait 


# CUDA_VISIBLE_DEVICES=2 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/frameworks/conrecon_L1.yaml --CONSTANTS#dataset="medmnist.OrganMNIST3D" --CONSTANTS#num_classes=11 &
CUDA_VISIBLE_DEVICES=0 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/frameworks/conrecon.yaml --CONSTANTS#dataset="medmnist.OrganMNIST3D" --CONSTANTS#num_classes=11 &
CUDA_VISIBLE_DEVICES=1 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/frameworks/reconstruction.yaml --CONSTANTS#dataset="medmnist.OrganMNIST3D" --CONSTANTS#num_classes=11 &


CUDA_VISIBLE_DEVICES=2 lighter fit --config_file=./experiments/evaluation/medmnist.yaml,./experiments/evaluation/frameworks/simclr.yaml --CONSTANTS#dataset="medmnist.OrganMNIST3D" --CONSTANTS#num_classes=11 &
