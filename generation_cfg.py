import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from copy import deepcopy
import soundfile as sf

from warping.utils import load_json, load_config
from warping.interpolant import DeterministicInterpolant
from warping.sit import prepare_sit_from_config
from warping.backbone import EncoderDecoder
from warping.sampling import denoise_latent


def main(args):
    output_dir = args.output_dir
    num_samples = args.num_samples
    num_steps = args.num_steps
    config_path = args.config_path
    stats_path = args.stats_path
    checkpoint_dir = args.checkpoint_dir
    cfg_strength = args.cfg_scale
    label = args.label

    label_mapping = {"maestro": 0,
                     "asap": 1}
    label = label_mapping[label]

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

    data_stats = load_json(stats_path)
    interpolant = DeterministicInterpolant(train_stats=data_stats,
                                           device=device)

    sit = prepare_sit_from_config(config=config,
                                  device=device)
    sit.load_state_dict(torch.load(checkpoint_path)[is_ema], strict=False)
    sit.eval()

    encdec = EncoderDecoder(device=device)

    os.makedirs(output_dir, exist_ok=True)

    y = torch.tensor([label] * batch_size, device=device)

    i = 0
    while i < num_samples:
        batch_size = min(batch_size, num_samples - i)

        x = denoise_latent(model=sit,
                           interpolant=interpolant,
                           seq_len=seq_len,
                           device=device,
                           y=y,
                           cfg_strength=cfg_strength,
                           batch_size=batch_size,
                           num_steps=num_steps)
        
        x = torch.transpose(x, 1, 2)
        audio_batch = encdec.decode(x)
        audio_batch = audio_batch.cpu().numpy()

        for j, sample in tqdm(enumerate(audio_batch)):
            filename = f"generated_{i + j}.mp3"
            filepath = os.path.join(output_dir, filename)
            sf.write(filepath,
                     sample,
                     44100)

        i += batch_size
        

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='directory to store generated samples')
    parser.add_argument('--num_samples', type=int, required=True, default=10, help='number of samples to generate')
    parser.add_argument('--num_steps', type=int, required=True, default=64, help='number of inference steps')
    parser.add_argument('--config_path', type=str, required=True, default=None, help='path to yaml config')
    parser.add_argument('--stats_path', type=str, required=True, help='path to dataset embedding statistics')    
    parser.add_argument('--checkpoint_dir', type=str, required=True, default=None, help='directory to store checkpoints')
    parser.add_argument('--cfg_scale', type=float, required=True, default=1, help='classifier-free guidance scale')
    parser.add_argument('--label', type=str, required=True, default=1, help='conditioning label')
    
    args = parser.parse_args()

    main(args)
