import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
from typing import Callable

from .interpolant import DeterministicInterpolant


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


class FlowTrainer:
    """Flow matching model trainer."""

    def __init__(self,
                 interpolant: DeterministicInterpolant,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: Optimizer,
                 num_epochs: int,
                 device: str,
                 scheduler: LRScheduler | None = None,
                 verbose: int = 5,
                 checkpoint_path: str | None = None,
                 metrics_logger: Callable | None = None):
        """Initializes an instance of FlowTrainer.
        
        Args:
            interpolant: Flow interpolant.
            train_loader: Train dataloader.
            val_loader: Validation dataloader.
            optimizer: Optimizer.
            num_epochs: The number of epochs.
            device: Device.
            scheduler: Learning rate scheduler.
            verbose: Logging frequency.
            checkpoint_path: The output path to model checkpoint.
            metrics_logger: Metrics logger.
        """
        self.interpolant = interpolant
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.scheduler = scheduler
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path
        self.metrics_logger = metrics_logger
    
    def loss_fn(self,
                model: nn.Module,
                x0: torch.Tensor,
                x1: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """Calculates loss.
        
        Args:
            model: Flow matching model.
            x0: Batch of samples at time 0 of shape (B, seq_len, D).
            x1: Batch of samples at time 1 of shape (B, seq_len, D).
            t: Batch of times of shape (B,).
        
        Returns:
            Loss value.
        """
        It = self.interpolant.xt(x0, x1, t)
        dtIt = self.interpolant.dtxt(x0, x1, t)

        bt = model(It, t)
        loss = F.mse_loss(bt, dtIt)
        return loss
    
    def _train_epoch(self,
                     model: nn.Module,
                     epoch: int,
                     ema: nn.Module | None = None) -> float:
        """One epoch pass."""
        epoch_loss = 0

        model.train()
        iters = len(self.train_loader)
        for i, x1 in enumerate(self.train_loader):
            x1 = x1.to(self.device)
            x0 = torch.randn_like(x1)
            t = torch.rand(len(x0)).to(self.device)

            loss = self.loss_fn(model=model,
                                x0=x0,
                                x1=x1,
                                t=t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.optimizer.step()
            if ema is not None:
                update_ema(ema, model)
            if self.scheduler is not None:
                self.scheduler.step(epoch + i / iters)
            self.optimizer.zero_grad()
            epoch_loss += loss.item()
        epoch_loss /= len(self.train_loader)
        return epoch_loss
    
    @torch.no_grad()
    def validate(self,
                 model: nn.Module) -> float:
        """Calculates loss on validation set."""
        val_loss = 0
        data_len = 0

        model.eval()
        for i, x1 in enumerate(self.val_loader):
            x1 = x1.to(self.device)
            x0 = torch.randn_like(x1)
            t = torch.rand(len(x0)).to(self.device)

            loss = self.loss_fn(model=model,
                                x0=x0,
                                x1=x1,
                                t=t)
            val_loss = val_loss + loss.item() * len(x1)
            data_len += len(x1)
        
        val_loss /= data_len
        return val_loss
    
    def train(self,
              model: nn.Module,
              ema: nn.Module | None = None):
        """Trains a flow matching model.
        
        Args:
            model: Flow matching model.
            ema: EMA version of a model (optional).
        """
        for i in tqdm(range(self.num_epochs)):
            train_loss = self._train_epoch(model=model,
                                           epoch=i,
                                           ema=ema)
            if ema is not None:
                val_loss = self.validate(ema)
            else:
                val_loss = self.validate(model)

            if i % self.verbose == 0:
                print(f"{i} Train loss: {train_loss}")
                print(f"{i} Validation loss: {val_loss}")
            
            if self.metrics_logger is not None:
                logging_dict = {"train_loss": train_loss,
                                "val_loss": val_loss}
                self.metrics_logger(logging_dict,
                                    epoch=i)

        print(f"Saving the model into {self.checkpoint_path}")
        checkpoint = {"model": model.state_dict()}
        if ema is not None:
            checkpoint["ema"] = ema.state_dict()
        torch.save(checkpoint, self.checkpoint_path)
