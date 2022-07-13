# Adapted/modfied from https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/ssl_finetuner.py
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

import torchmetrics

import math
from functools import partial

class SSLFineTuner(LightningModule):
    """
    Finetunes a self-supervised learning backbone using the standard evaluation protocol of a singler layer MLP,
    or using MLP head
    """

    def __init__(
        self,
        params: dict,
        backbone: torch.nn.Module,
        in_features: int = 2048,
        num_classes: int = 1000,
    ):
        """
        Args:
            backbone: a pretrained model
            in_features: feature dim of backbone outputs
            num_classes: classes of the dataset
            hidden_dim: dim of the MLP (1024 default used in self-supervised literature)
        """
        super().__init__()
        self.params = params      
        self.in_features = in_features
        self.num_classes = num_classes
        self.backbone = backbone

        self.finetune = self.params.get("finetune", False)
        self.prediction_type = self.params.get("prediction_type", "regression")
        self.hidden_dim = self.params.get("hidden_dim", None)

        self.learning_rate = self.params.get("learning_rate", 0.1)
        self.learning_rate_backbone = self.params.get("learning_rate_backbone", 0.01)
        self.nesterov = self.params.get("nesterov", False)
        self.weight_decay = self.params.get("weight_decay", 1e-5)

        self.optimizer = self.params.get("optimizer", 'Adam')
        self.scheduler = self.params.get('scheduler', 'CosineAnnealingLR')
        self.T_max = self.params.get('T_max', 100)

        self.scheduler_params = self.hparams.get(
            'scheduler_params',
            {
                'T_max': self.T_max,
                'eta_min': 0,
             }
        )           

        self.decay_epochs = self.params.get("decay_epochs", (60,80))
        self.gamma = self.params.get("gamma", 0.1)
        self.epochs = self.params.get("epochs", 100)
        self.final_lr = self.params.get("final_lr", 0.0)
        self.dropout = self.params.get("dropout", 0.0)

        self.linear_layer = SSLEvaluator(
            n_input=self.in_features,
            n_classes=self.num_classes,
            p=self.dropout,
            n_hidden=self.hidden_dim,
        )
        if self.params['verbose']:
            print(self.linear_layer)
            
        metric_collection = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(),
            torchmetrics.Precision(),
            torchmetrics.Recall(),
            torchmetrics.F1Score(),
            torchmetrics.Specificity()])

        self.train_metrics = metric_collection.clone(prefix='train_')
        self.val_metrics = metric_collection.clone(prefix='val_')
        self.test_metrics = metric_collection.clone(prefix='test_')

        self.hparams.update(params)
        self.save_hyperparameters()


    def forward(self, x):#batch):
        """
        Forward needs batch, not just x, as trainer.predict passes batch
        """

        #x, _ = batch
        
        if not self.finetune:
            with torch.no_grad():
                feats = self.backbone(x)
        else:
            feats = self.backbone(x)
            
        feats  = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)

        return logits
    
    def on_train_epoch_start(self) -> None:

        if not self.finetune:
            self.backbone.eval()

    def log_metrics(self, pred, target, stage='train'):

        if stage=='train':
            metrics = self.train_metrics
        if stage=='val':
            metrics = self.val_metrics
        if stage=='test':
            metrics = self.test_metrics
            
        if self.prediction_type.upper() == 'CLASSIFICATION':

            if self.num_classes > 1:
                pred = pred.softmax(-1)
            else:
                pred = torch.sigmoid(pred).view(-1).round().int()
                target = target.round().int()

        if self.prediction_type.upper() == 'REGRESSION':

            pred = pred.view(-1).round().int()
            target = target.round().int()

        output = metrics(pred, target)
        # use log_dict instead of log
        # metrics are logged with keys: train_Accuracy, train_Precision and train_Recall
        self.log_dict(output)

        # for metric in metrics:
        #     metrics[metric](pred, target)
        #     self.log(f"{stage}_{metric}", metrics[metric])

        
    def training_step(self, batch, batch_idx):

        loss, pred, target = self.shared_step(batch)
        # self.log_metrics(pred, target, 'train')
        self.log("train_loss", loss, prog_bar=True)
        self.log_metrics(pred, target, stage='train')

        return loss
        
        # return {'loss': loss, 'pred': pred, 'target': target}
    # def training_step_end(self, outputs):
    #     #update and log
    #     self.log_metrics(outputs['pred'], outputs['target'], stage='train')

    def validation_step(self, batch, batch_idx):

        loss, pred, target = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log_metrics(pred, target, stage='val')

        return loss

    # def validation_step_end(self, outputs):
    #     #update and log
    #     self.log_metrics(outputs['pred'], outputs['target'], stage='val')

    def test_step(self, batch, batch_idx):
        loss, pred, target = self.shared_step(batch)
        self.log("test_loss", loss, sync_dist=True)
        self.log_metrics(pred, target, stage='test')

        return loss

    # def test_step_end(self, outputs):
    #     #update and log
    #     self.log_metrics(outputs['pred'], outputs['target'], stage='test')

    def shared_step(self, batch):

        data, target = batch

        #logits = self(batch)
        pred = self(data)

        # print(feats.shape, logits.view(-1).shape, x.shape, y.shape)
        
        if self.num_classes > 1:
            loss = F.cross_entropy(pred, target)
        else:
            if self.prediction_type.upper() == 'CLASSIFICATION':
                loss = torch.nn.BCEWithLogitsLoss()(pred.view(-1), target.float())
                
            elif self.prediction_type.upper() == 'REGRESSION': 
                loss = torch.nn.MSELoss()(pred.view(-1), target.float())

        return loss, pred, target

    def configure_optimizers(self):

        optimizer = getattr(torch.optim, self.optimizer)

        if not self.finetune:
            parameters = self.linear_layer.parameters()
            # optimizer = optimizer(parameters, self.learning_rate)

        else:
            # Give different learning rates to backbone and classification head,
            # As pretrained backbone likely requires smaller updates
            parameters = [
                {"params": self.backbone.parameters(), 'lr': self.learning_rate_backbone},
                {"params": self.linear_layer.parameters(), 'lr': self.learning_rate},
                ]

        optimizer = optimizer(parameters, self.learning_rate)
        
        scheduler = partial(getattr(torch.optim.lr_scheduler, self.scheduler), optimizer)
        scheduler = scheduler(**self.scheduler_params)

        return [optimizer], [scheduler]

# Copied from https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/evaluator.py
class SSLEvaluator(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden

        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input,
                          n_classes,
                          bias=True,
                         ),
            )
            
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True),
            )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)
    
