CUDA_VISIBLE_DEVICES=0 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_segresnet_encoder.yaml --vars#dataset="medmnist.NoduleMNIST3D" --vars#num_classes=2 &
CUDA_VISIBLE_DEVICES=1 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_unet_encoder.yaml --vars#dataset="medmnist.NoduleMNIST3D" --vars#num_classes=2 &

CUDA_VISIBLE_DEVICES=2 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_segresnet_encoder.yaml --vars#dataset="medmnist.FractureMNIST3D" --vars#num_classes=3 &
CUDA_VISIBLE_DEVICES=3 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_unet_encoder.yaml --vars#dataset="medmnist.FractureMNIST3D" --vars#num_classes=3 &
wait

CUDA_VISIBLE_DEVICES=0 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_segresnet_encoder.yaml --vars#dataset="medmnist.OrganMNIST3D" --vars#num_classes=11 &
CUDA_VISIBLE_DEVICES=1 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_unet_encoder.yaml --vars#dataset="medmnist.OrganMNIST3D" --vars#num_classes=11 &


# CUDA_VISIBLE_DEVICES=0 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/frameworks/conrecon_L1.yaml --vars#dataset="medmnist.NoduleMNIST3D" --vars#num_classes=2 &
CUDA_VISIBLE_DEVICES=2 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/frameworks/conrecon.yaml --vars#dataset="medmnist.NoduleMNIST3D" --vars#num_classes=2 &
CUDA_VISIBLE_DEVICES=3 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/frameworks/reconstruction.yaml --vars#dataset="medmnist.NoduleMNIST3D" --vars#num_classes=2 &
wait 
CUDA_VISIBLE_DEVICES=0 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/frameworks/simclr.yaml --vars#dataset="medmnist.NoduleMNIST3D" --vars#num_classes=2 &


# CUDA_VISIBLE_DEVICES=1 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/frameworks/conrecon_L1.yaml --vars#dataset="medmnist.FractureMNIST3D" --vars#num_classes=3 &
CUDA_VISIBLE_DEVICES=1 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/frameworks/conrecon.yaml --vars#dataset="medmnist.FractureMNIST3D" --vars#num_classes=3 &
CUDA_VISIBLE_DEVICES=2 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/frameworks/reconstruction.yaml --vars#dataset="medmnist.FractureMNIST3D" --vars#num_classes=3 &
CUDA_VISIBLE_DEVICES=3 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/frameworks/simclr.yaml --vars#dataset="medmnist.FractureMNIST3D" --vars#num_classes=3 &

wait 


# CUDA_VISIBLE_DEVICES=2 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/frameworks/conrecon_L1.yaml --vars#dataset="medmnist.OrganMNIST3D" --vars#num_classes=11 &
CUDA_VISIBLE_DEVICES=0 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/frameworks/conrecon.yaml --vars#dataset="medmnist.OrganMNIST3D" --vars#num_classes=11 &
CUDA_VISIBLE_DEVICES=1 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/frameworks/reconstruction.yaml --vars#dataset="medmnist.OrganMNIST3D" --vars#num_classes=11 &


CUDA_VISIBLE_DEVICES=2 lighter fit --config=./evaluation/medmnist.yaml,./evaluation/frameworks/simclr.yaml --vars#dataset="medmnist.OrganMNIST3D" --vars#num_classes=11 &
