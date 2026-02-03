import os
import argparse
import comet_ml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from copy import deepcopy

from warping.utils import load_json, prepare_filepaths_from_metadata, \
                          load_config
from warping.data import EmbeddingDataset
from warping.interpolant import DeterministicInterpolant
from warping.sit import prepare_sit_from_config
from warping.trainer import FlowTrainer, update_ema, requires_grad


def main(args):
    metadata_path = args.metadata_path
    stats_path = args.stats_path
    project_name = args.project_name
    config_path = args.config_path
    checkpoint_dir = args.checkpoint_dir

    config = load_config(config_path)

    if project_name is not None:
        comet_ml.login()
        exp = comet_ml.start(project_name=project_name)
        metrics_logger = exp.log_metrics
    else:
        metrics_logger = None

    batch_size = config['data']['batch_size']
    seq_len = config['model']['seq_len']
    checkpoint_name = config['model']['checkpoint_name']
    learning_rate = config['train']['lr']
    weight_decay = config['train']['weight_decay']
    num_epochs = config['train']['num_epochs']
    is_ema = config['train']['is_ema']

    checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metadata = load_json(metadata_path)
    train_dict_lst = prepare_filepaths_from_metadata(metadata,
                                                     split='train')
    val_dict_lst = prepare_filepaths_from_metadata(metadata,
                                                   split='validation')

    train_dataset = EmbeddingDataset(emb_dict_lst=train_dict_lst,
                                     is_cache=True,
                                     seq_len=seq_len)
    val_dataset = EmbeddingDataset(emb_dict_lst=val_dict_lst,
                                   is_cache=True,
                                   seq_len=seq_len)
    
    #Cache datasets
    for x in train_dataset:
        pass

    for x in val_dataset:
        pass

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False)
    
    data_stats = load_json(stats_path)
    interpolant = DeterministicInterpolant(train_stats=data_stats,
                                           device=device)

    sit = prepare_sit_from_config(config=config,
                                  device=device)

    if is_ema:
        ema = deepcopy(sit).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        update_ema(ema, sit, decay=0)  # Ensure EMA is initialized with synced weights
        ema.eval() # EMA model should always be in eval mode
    else:
        ema = None

    optimizer = torch.optim.AdamW(sit.parameters(),
                                  lr=learning_rate,
                                  weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer,
                                            num_epochs * len(train_loader),
                                            eta_min=0)
    trainer = FlowTrainer(interpolant=interpolant,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          optimizer=optimizer,
                          num_epochs=num_epochs,
                          device=device,
                          scheduler=scheduler,
                          verbose=50,
                          checkpoint_path=checkpoint_path,
                          metrics_logger=metrics_logger)
    trainer.train(sit,
                  ema=ema)

    if project_name is not None:
        exp.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', type=str, required=True, help='path to audio dataset metadata')
    parser.add_argument('--stats_path', type=str, required=True, help='path to dataset embedding statistics')
    parser.add_argument('--project_name', type=str, required=False, default=None, help='wandb project name')
    parser.add_argument('--config_path', type=str, required=True, default=None, help='path to yaml config')
    parser.add_argument('--checkpoint_dir', type=str, required=True, default=None, help='directory to store checkpoints')
    
    args = parser.parse_args()

    main(args)
