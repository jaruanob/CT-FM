#### CT-FM #####
lighter fit --config=./evaluation/medmnist.yaml --vars#dataset="medmnist.NoduleMNIST3D" --vars#num_classes=2 --system#model#trunk#ckpt_path="/media/volume/CT-RATE/CT-FM/latest_checkpoints/epoch=449-step=225000-v1.ckpt" --vars#name="ct_fm" --vars#project="medmnist"
lighter fit --config=./evaluation/medmnist.yaml --vars#dataset="medmnist.FractureMNIST3D" --vars#num_classes=3 --system#model#trunk#ckpt_path="/media/volume/CT-RATE/CT-FM/latest_checkpoints/epoch=449-step=225000-v1.ckpt" --vars#name="ct_fm" --vars#project="medmnist"
lighter fit --config=./evaluation/medmnist.yaml --vars#dataset="medmnist.OrganMNIST3D" --vars#num_classes=11 --system#model#trunk#ckpt_path="/media/volume/CT-RATE/CT-FM/latest_checkpoints/epoch=449-step=225000-v1.ckpt" --vars#name="ct_fm" --vars#project="medmnist"

### Suprem Unet ####
lighter fit --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_unetencoder.yaml --vars#dataset="medmnist.NoduleMNIST3D" --vars#num_classes=2 --vars#name="suprem_unet" --vars#project="medmnist"
lighter fit --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_unetencoder.yaml --vars#dataset="medmnist.FractureMNIST3D" --vars#num_classes=3 --vars#name="suprem_unet" --vars#project="medmnist"
lighter fit --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_unetencoder.yaml --vars#dataset="medmnist.OrganMNIST3D" --vars#num_classes=11 --vars#name="suprem_unet" --vars#project="medmnist"

### Suprem Unet ####
lighter fit --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_segresencoder.yaml --vars#dataset="medmnist.NoduleMNIST3D" --vars#num_classes=2 --vars#name="suprem_segresnet" --vars#project="medmnist"
lighter fit --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_segresencoder.yaml --vars#dataset="medmnist.FractureMNIST3D" --vars#num_classes=3 --vars#name="suprem_segresnet" --vars#project="medmnist"
lighter fit --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_segresencoder.yaml --vars#dataset="medmnist.OrganMNIST3D" --vars#num_classes=11 --vars#name="suprem_segresnet" --vars#project="medmnist"

### VISTA3D ####
lighter fit --config=./evaluation/medmnist.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#dataset="medmnist.NoduleMNIST3D" --vars#num_classes=2 --vars#name="vista3d" --vars#project="medmnist" 
lighter fit --config=./evaluation/medmnist.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#dataset="medmnist.FractureMNIST3D" --vars#num_classes=3 --vars#name="vista3d" --vars#project="medmnist" 
lighter fit --config=./evaluation/medmnist.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#dataset="medmnist.OrganMNIST3D" --vars#num_classes=11 --vars#name="vista3d" --vars#project="medmnist" 



# ### Testing ####
# lighter test --config=./evaluation/medmnist.yaml --vars#dataset="medmnist.NoduleMNIST3D" --vars#num_classes=2 --system#model#trunk#ckpt_path="/media/volume/CT-RATE/CT-FM/latest_checkpoints/epoch=449-step=225000-v1.ckpt" --vars#name="ct_fm" --vars#project="medmnist"
# lighter test --config=./evaluation/medmnist.yaml --vars#dataset="medmnist.FractureMNIST3D" --vars#num_classes=3 --system#model#trunk#ckpt_path="/media/volume/CT-RATE/CT-FM/latest_checkpoints/epoch=449-step=225000-v1.ckpt" --vars#name="ct_fm" --vars#project="medmnist"
# lighter test --config=./evaluation/medmnist.yaml --vars#dataset="medmnist.OrganMNIST3D" --vars#num_classes=11 --system#model#trunk#ckpt_path="/media/volume/CT-RATE/CT-FM/latest_checkpoints/epoch=449-step=225000-v1.ckpt" --vars#name="ct_fm" --vars#project="medmnist"

# lighter test --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_unetencoder.yaml --vars#dataset="medmnist.NoduleMNIST3D" --vars#num_classes=2 --vars#name="suprem_unet" --vars#project="medmnist"
# lighter test --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_unetencoder.yaml --vars#dataset="medmnist.FractureMNIST3D" --vars#num_classes=3 --vars#name="suprem_unet" --vars#project="medmnist"
# lighter test --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_unetencoder.yaml --vars#dataset="medmnist.OrganMNIST3D" --vars#num_classes=11 --vars#name="suprem_unet" --vars#project="medmnist"

# lighter test --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_segresencoder.yaml --vars#dataset="medmnist.NoduleMNIST3D" --vars#num_classes=2 --vars#name="suprem_segresnet" --vars#project="medmnist"
# lighter test --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_segresencoder.yaml --vars#dataset="medmnist.FractureMNIST3D" --vars#num_classes=3 --vars#name="suprem_segresnet" --vars#project="medmnist"
# lighter test --config=./evaluation/medmnist.yaml,./evaluation/baselines/suprem_segresencoder.yaml --vars#dataset="medmnist.OrganMNIST3D" --vars#num_classes=11 --vars#name="suprem_segresnet" --vars#project="medmnist"

# lighter test --config=./evaluation/medmnist.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#dataset="medmnist.NoduleMNIST3D" --vars#num_classes=2 --vars#name="vista3d" --vars#project="medmnist" 
# lighter test --config=./evaluation/medmnist.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#dataset="medmnist.FractureMNIST3D" --vars#num_classes=3 --vars#name="vista3d" --vars#project="medmnist" 
# lighter test --config=./evaluation/medmnist.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#dataset="medmnist.OrganMNIST3D" --vars#num_classes=11 --vars#name="vista3d" --vars#project="medmnist" 
