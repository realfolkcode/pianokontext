import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from copy import deepcopy
from torch.utils.data import DataLoader
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
                                 seq_len=seq_len,
                                 is_from_start=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False)

    for sample in tqdm(val_dataset):
        pass

    t_space = torch.linspace(0, 1, num_steps + 1)[:-1].to(device)

    os.makedirs(output_dir, exist_ok=True)

    gt_dir = os.path.join(output_dir, "groundtruth")
    rec_dir = os.path.join(output_dir, "reconstruction")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)

    for score_id, batch in tqdm(enumerate(val_loader)):
        context = batch["deadpan"].to(device)
        context = torch.cat(num_samples * [context])
        context_mask = batch["deadpan_mask"].to(device)
        context_mask = torch.cat(num_samples * [context_mask])

        x_gt = batch["expressive"].to(device)
        x_gt_mask = batch["expressive_mask"].to(device)

        x_init = torch.randn_like(context)
        x_init_mask = x_gt_mask.clone()
        x_init_mask = torch.cat(num_samples * [x_init_mask])

        x = solve_ode_flux(flux,
                           x_init=x_init,
                           x_init_mask=x_init_mask,
                           time_grid=t_space,
                           context=context,
                           context_mask=context_mask)

        out_score_dir = os.path.join(rec_dir, f"score_{score_id}")
        os.makedirs(out_score_dir, exist_ok=True)
        for i in range(len(x)):
            audio = encdec.decode(x[i][x_init_mask[i]].T.to(device))
            audio = audio.cpu().numpy()
            output_path = os.path.join(out_score_dir, f"rec_{i}.mp3")
            sf.write(output_path, audio.T, 44100)

        out_score_dir = os.path.join(gt_dir, f"score_{score_id}")
        os.makedirs(out_score_dir, exist_ok=True)
        audio = encdec.decode(x_gt[x_gt_mask].T.to(device))
        audio = audio.cpu().numpy()
        output_path = os.path.join(out_score_dir, "gt.mp3")
        sf.write(output_path, audio.T, 44100)
        

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
