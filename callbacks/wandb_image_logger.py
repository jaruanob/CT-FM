from typing import Any, Callable, Dict, Union

import gc
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path


import torch
from loguru import logger
from pytorch_lightning import Callback, Trainer

import wandb
from lighter import LighterSystem

class WandbImageLogger(Callback):
    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LighterSystem, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ):
        """Called when the validation batch ends."""
        if batch_idx == 0:
            x = batch["input"].squeeze()
            y = batch["target"].squeeze()
            y_pred = outputs["pred"].squeeze()

            assert x.shape == y.shape == y_pred.shape

            data = []
            for x_slice, y_slice, y_pred_slice in zip(x, y, y_pred):
                x_slice = x_slice.cpu().numpy()
                y_slice = y_slice.cpu().numpy()
                y_pred_slice = y_pred_slice.cpu().numpy()

                image = wandb.Image(x_slice, masks={
                    "predictions": {"mask_data": y_pred_slice},
                    "ground_truth": {"mask_data": y_slice},
                })

                data.append([image])
            
            trainer.logger.log_table("pred_table", ["image"], data)
