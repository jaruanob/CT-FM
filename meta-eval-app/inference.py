import streamlit as st
import monai
from model import load_model
from utils import IterableDataset
import torch


def run(count, patch_size):
    """
    Run the CT-FM algorithm on the selected scans and points.

    Args:
        count (int): Number of scans to process.
        patch_size (tuple): Size of the patch to extract around the selected point.
    """
    # Get the selected images and points from the session state
    images = [st.session_state[f"data_{idx}"]["image"] for idx in range(count)]
    points = [st.session_state[f"point_{idx}"] for idx in range(count)]
    
    # Find the index of the reference image (the one with a selected point)
    ref_idx = next((idx for idx in range(count) if points[idx] is not None), None)
    ref_image = images[ref_idx]
    ref_point = points[ref_idx]

    # Crop and pad the reference patch around the selected point
    cropper = monai.transforms.Crop()
    padder = monai.transforms.SpatialPad(spatial_size=patch_size, mode="constant")
    slices = cropper.compute_slices(roi_center=ref_point, roi_size=patch_size)
    ref_patch = cropper(ref_image, slices)
    ref_patch = padder(ref_patch)

    # Set up the sliding window splitter for patch extraction
    
    with torch.no_grad():
        # Load the pre-trained model
        model = load_model()
        sim_fn = torch.nn.CosineSimilarity()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ref_patch = ref_patch.to(device)
        
        # Compute the embedding for the reference patch
        ref_embedding = model(ref_patch.unsqueeze(0))

        def predictor(x):
            """
            Predict the distance between a patch and the reference patch.

            Args:
                x (torch.Tensor): Input patch tensor.

            Returns:
                torch.Tensor: Distance between the input patch and the reference patch.
            """
            x = x.to(device)
            pred = model(x)
            sim = sim_fn(pred, ref_embedding)
            return sim

        matching_points = []
        for idx, moving_image in enumerate(images):
            if idx == ref_idx:
                # Skip the reference image and use the selected point
                matching_points.append(ref_point)
                continue

            similarity = []
            
            splitter = monai.inferers.SlidingWindowSplitter(patch_size, 0.25)
            dataset = IterableDataset(splitter(moving_image.unsqueeze(0)))
            patch_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

            for patch, location in patch_dataloader:
                sim = predictor(patch.squeeze(dim=1))

                for d, z, y, x in zip(sim, location[0]+patch_size[0]//2, location[1]+patch_size[1]//2, location[2]+patch_size[2]//2):
                    similarity.append({
                        "sim": d.item(),
                        "location": (z.item(), y.item(), x.item()) 
                    })

            print(len(similarity))
            sorted_sim = sorted(similarity, key=lambda x: x["sim"])
            print(sorted_sim[0], sorted_sim[-1])
            max_sim = sorted_sim[-1]
            # Find the patch with the minimum distance to the reference patch
            matching_points.append(max_sim["location"])

    for idx, point in enumerate(matching_points):
        st.session_state[f"data_{idx}"]["bbox"] = ((point[0] - patch_size[0]//2, point[1] - patch_size[1]//2, point[2] - patch_size[2]//2), (point[0] + patch_size[0]//2, point[1] + patch_size[1]//2, point[2] + patch_size[2]//2))
        st.session_state[f"point_{idx}"] = point
    st.session_state.finished = True
