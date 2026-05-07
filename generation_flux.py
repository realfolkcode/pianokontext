import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from copy import deepcopy
import soundfile as sf

from warping.utils import load_json, load_config, prepare_filepaths_from_aligned_metadata
from warping.interpolant import DeterministicInterpolant
from warping.flux import prepare_flux_from_config
from warping.backbone import EncoderDecoder
from warping.sampling import solve_ode_flux
from warping.data import AlignedDataset


def main(args):
    metadata_path = args.metadata_path
    output_dir = args.output_dir
    num_samples = args.num_samples
    num_steps = args.num_steps
    config_path = args.config_path
    checkpoint_dir = args.checkpoint_dir

    config = load_config(config_path)

    batch_size = config['data']['batch_size']
    seq_len = config['model']['seq_len']
    checkpoint_name = config['model']['checkpoint_name']
    is_ema = config['train']['is_ema']

    checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
    if is_ema:
        is_ema = "ema"
    else:
        is_ema = "model"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_stats = None
    interpolant = DeterministicInterpolant(train_stats=data_stats,
                                           device=device)

    flux = prepare_flux_from_config(config=config,
                                    device=device)
    flux.load_state_dict(torch.load(checkpoint_path)[is_ema], strict=False)
    flux.eval()

    encdec = EncoderDecoder(device=device)

    metadata = load_json(metadata_path)
    val_dict_lst = prepare_filepaths_from_aligned_metadata(metadata,
                                                           split='validation')
    val_dataset = AlignedDataset(emb_dict_lst=val_dict_lst,
                                 is_cache=True,
                                 seq_len=seq_len)

    os.makedirs(output_dir, exist_ok=True)
        

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', type=str, required=True, help='path to audio dataset metadata')
    parser.add_argument('--output_dir', type=str, required=True, help='directory to store generated samples')
    parser.add_argument('--num_samples', type=int, required=True, default=5, help='number of samples to generate')
    parser.add_argument('--num_steps', type=int, required=True, default=64, help='number of inference steps')
    parser.add_argument('--config_path', type=str, required=True, default=None, help='path to yaml config')  
    parser.add_argument('--checkpoint_dir', type=str, required=True, default=None, help='directory to store checkpoints')
    
    args = parser.parse_args()

    main(args)
