import streamlit as st
import monai
from model import load_model
from utils import IterableDataset
import torch


def run(count):
    """
    Run the CT-FM algorithm on the selected scans and points.

    Args:
        count (int): Number of scans to process.
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
    padder = monai.transforms.SpatialPad(spatial_size=(64, 128, 128), mode="constant")
    slices = cropper.compute_slices(roi_center=ref_point, roi_size=(64, 128, 128))
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
            
            splitter = monai.inferers.SlidingWindowSplitter((64, 128, 128), 0.1)
            dataset = IterableDataset(splitter(moving_image.unsqueeze(0)))
            patch_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

            for patch, location in patch_dataloader:
                sim = predictor(patch.squeeze(dim=1))

                for d, z, y, x in zip(sim, location[0]+32, location[1]+64, location[2]+64):
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

    # Crop patches around the matching points and update the session state
    for idx, point in enumerate(matching_points):
        slices = cropper.compute_slices(roi_center=point, roi_size=(64, 128, 128))
        patch = cropper(images[idx], slices)
        st.session_state[f"data_{idx}"]["image"] = patch
            
    st.session_state.finished = True
