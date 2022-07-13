from typing import Any, Optional, Sequence, List

import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter

import os

class OutputExtractor(pl.LightningModule):
    """
    Pass data through network to extract model outputs
    """
    def __init__(
        self,
        backbone: torch.nn.Module,
    ):    
        super(OutputExtractor, self).__init__()

        # pass
        self.backbone = backbone
        self.backbone.eval()

    def forward(self, batch):
        
        x, _ = batch
        
        z_emb = self.backbone(x)
       
        return z_emb
    
    def predict(self, batch, batch_idx: int, dataloader_idx: int=None):
        return self(batch)

class OutputWriter(BasePredictionWriter):
    """
    Special callback to write model outputs to specified locations.
    
    model outputs are written out batch-by-batch into seperate files,
    which need to be concatenated after the fact for easier use.
    
    Representation creation (usually) requires gpu, while concatentation does not
    """
    def __init__(
        self,
        output_dir: str,
        file_head: str = 'output',
        overwrite: bool = False,
        write_interval: str = "batch",
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.file_head = file_head 
        self.overwrite = overwrite

    def write_on_batch_end(
        self,
        trainer,
        pl_module: 'LightningModule',
        prediction: Any,
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int
    ):
        
        batch_file_out = f"{self.file_head}_batch_{batch_idx:07d}"
        num_gpus=0,
        if prediction.is_cuda:
            current_device = torch.cuda.current_device()
            num_gpus=torch.cuda.device_count()
            batch_file_out += f"_{current_device:03}"    
        
        file_out = os.path.join(self.output_dir, batch_file_out+".npz")
        if not os.path.isfile(file_out) or self.overwrite:
            # if prediction.is_cuda:
            #     prediction = prediction.cpu()

            np.savez(
                file_out,
                data=prediction.cpu().numpy() if prediction.is_cuda else prediction.numpy(),
                batch_indices=batch_indices,
                num_gpus=num_gpus,
            )

        prediction, batch_indices = [], []
        
    def write_on_epoch_end(
        self,
        trainer,
        pl_module: 'LightningModule',
        predictions: List[Any],
        batch_indices: List[Any]
    ):
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))
        
