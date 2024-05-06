from PIL import Image, ImageEnhance, ImageDraw
import streamlit as st
import numpy as np
import torch
from torch.nn import Module
from typing import Dict, List
from loguru import logger


def make_fig(image, preds, point_axs=None, current_idx=None, view=None):
    # Convert A to an image
    image = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # Create a yellow mask from B
    if preds is not None:
        mask = np.where(preds == 1, 255, 0).astype(np.uint8)
        mask = Image.merge("RGB", 
                           (Image.fromarray(mask), 
                            Image.fromarray(mask), 
                            Image.fromarray(np.zeros_like(mask, dtype=np.uint8))))

        # Overlay the mask on the image
        image = Image.blend(image.convert("RGB"), mask, alpha=st.session_state.transparency)
    
    if point_axs is not None:
        draw = ImageDraw.Draw(image)
        radius = 10
        z, y, x = point_axs
        if z == current_idx:
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="blue")
    return image


def adjust_prefix_and_load_state_dict(
    model: Module, ckpt_path: str, ckpt_to_model_prefix: Dict[str, str] = None, layers_to_ignore: List[str] = None
) -> Module:
    """Load state_dict from a checkpoint into a model using `torch.load(strict=False`).
    `ckpt_to_model_prefix` mapping allows to rename the prefix of the checkpoint's state_dict keys
    so that they match those of the model's state_dict. This is often needed when a model was trained
    as a backbone of another model, so its state_dict keys won't be the same to those of a standalone
    version of that model. Prior to defining the `ckpt_to_model_prefix`, it is advised to manually check
    for mismatch between the names and specify them accordingly.

    Args:
        model (Module): PyTorch model instance to load the state_dict into.
        ckpt_path (str): Path to the checkpoint.
        ckpt_to_model_prefix (Dict[str, str], optional): A dictionary that maps keys in the checkpoint's
            state_dict to keys in the model's state_dict. If None, no key mapping is performed. Defaults to None.
        layers_to_ignore (List[str], optional): A list of layer names that won't be loaded into the model.
            Specify the names as they are after `ckpt_to_model_prefix` is applied. Defaults to None.
    Returns:
        The model instance with the state_dict loaded.

    Raises:
        ValueError: If there is no overlap between checkpoint's and model's state_dict.
    """

    # Load checkpoint
    ckpt = torch.load(ckpt_path)

    # Check if the checkpoint is a model's state_dict or a LighterSystem checkpoint.
    # A LighterSystem checkpoint contains the model’s entire internal state, we only need its state_dict.
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
        # Remove the "model." prefix from the checkpoint's state_dict keys. This is characteristic to LighterSystem.
        ckpt = {key.replace("model.", ""): value for key, value in ckpt.items()}

    # Adjust the keys in the checkpoint's state_dict to match the the model's state_dict's keys.
    if ckpt_to_model_prefix is not None:
        for ckpt_prefix, model_prefix in ckpt_to_model_prefix.items():
            # Add a dot at the end of the prefix if necessary.
            ckpt_prefix = ckpt_prefix if ckpt_prefix == "" or ckpt_prefix.endswith(".") else f"{ckpt_prefix}."
            model_prefix = model_prefix if model_prefix == "" or model_prefix.endswith(".") else f"{model_prefix}."
            if ckpt_prefix != "":
                # Replace ckpt_prefix with model_prefix in the checkpoint state_dict
                ckpt = {key.replace(ckpt_prefix, model_prefix): value for key, value in ckpt.items() if ckpt_prefix in key}
            else:
                # Add the model_prefix before the current key name if there's no specific ckpt_prefix
                ckpt = {f"{model_prefix}{key}": value for key, value in ckpt.items() if ckpt_prefix in key}

    # Check if there is no overlap between the checkpoint's and model's state_dict.
    if not set(ckpt.keys()) & set(model.state_dict().keys()):
        raise ValueError(
            "There is no overlap between checkpoint's and model's state_dict. Check their "
            "`state_dict` keys and adjust accordingly using `ckpt_prefix` and `model_prefix`."
        )

    # Remove the layers that are not to be loaded.
    if layers_to_ignore is not None:
        for layer in layers_to_ignore:
            ckpt.pop(layer)

    # Load the adjusted state_dict into the model instance.
    incompatible_keys = model.load_state_dict(ckpt, strict=False)

    # Log the incompatible keys during checkpoint loading.
    if len(incompatible_keys.missing_keys) > 0 or len(incompatible_keys.unexpected_keys) > 0:
        logger.info(f"Encountered incompatible keys during checkpoint loading. If intended, ignore.\n{incompatible_keys}")
    else:
        logger.info("Checkpoint loaded successfully.")

    return model


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator