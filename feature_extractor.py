# Imports
import torch
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
from lighter_zoo import SegResEncoder
from monai.transforms import (
    Compose, LoadImage, EnsureType, Orientation,
    ScaleIntensityRange, CropForeground
)
from monai.inferers import SlidingWindowInferer
# import nibabel as nib
# import numpy as np
# Load pre-trained model
# model = SegResEncoder.from_pretrained(
#     "project-lighter/ct_fm_feature_extractor"
# )
# model.eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


model = torch.load('checkpoints/ct_fm_feature_extractor.pth')
model.eval()
#model = model.to('cuda')

# Preprocessing pipeline
preprocess = Compose([
    LoadImage(ensure_channel_first=True),  # Load image and ensure channel dimension
    EnsureType(),                         # Ensure correct data type
    Orientation(axcodes="SPL"),           # Standardize orientation
    # Scale intensity to [0,1] range, clipping outliers
    ScaleIntensityRange(
        a_min=-1024,    # Min HU value
        a_max=2048,     # Max HU value
        b_min=0,        # Target min
        b_max=1,        # Target max
        clip=True       # Clip values outside range
    ),
    CropForeground()    # Remove background to reduce computation
])

# Load the image (this gives you a torch tensor and metadata)

# Input path
input_path = "/home/vlab/Documents/Collections/Dental/STS-Tooth/SD-Tooth/STS-3D-Tooth/Integrity/Labeled/Image/Integrity_L_004.nii.gz"

# Preprocess input
input_tensor = preprocess(input_path)#.to('cuda')

# loader = LoadImage(ensure_channel_first=True)
# image_tensor = loader(input_path)

# # Compute min and max HU values
# a_min = torch.min(image_tensor).item()
# a_max = torch.max(image_tensor).item()

# print(f"Min HU (a_min): {a_min:.2f}")
# print(f"Max HU (a_max): {a_max:.2f}")

# nii = nib.load(input_path)
# data = nii.get_fdata()  # raw voxel values

# print("Data shape:", data.shape)
# print("Data type:", data.dtype)
# print("Contains NaN?", np.isnan(data).any())
# print("Contains Inf?", np.isinf(data).any())

# # Check if scaling (slope/intercept) is needed
# slope = nii.header.get('scl_slope', 1.0)
# intercept = nii.header.get('scl_inter', 0.0)
# hu_data = data * slope + intercept

# a_min = np.nanmin(hu_data)
# a_max = np.nanmax(hu_data)

# print(f"Real Min HU: {a_min:.2f}")
# print(f"Real Max HU: {a_max:.2f}")

# quit()

# Run inference
with torch.no_grad():
    output = model(input_tensor.unsqueeze(0))[-1]

    # Average pooling compressed the feature vector across all patches. If this is not desired, remove this line and 
    # use the output tensor directly which will give you the feature maps in a low-dimensional space.
    avg_output = torch.nn.functional.adaptive_avg_pool3d(output, 1).squeeze()

print("✅ Feature extraction completed")
print(f"Output shape: {avg_output.shape}")



# Plot distribution of features
import matplotlib.pyplot as plt
_ = plt.hist(avg_output.cpu().numpy(), bins=100)
plt.show()