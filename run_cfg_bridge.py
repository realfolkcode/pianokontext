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
from warping.sit import prepare_sit_from_config
from warping.backbone import EncoderDecoder
from warping.sampling import CFGBridge
from warping.data import EmbeddingDataset


def main(args):
    metadata_path = args.metadata_path
    output_dir = args.output_dir
    num_samples = args.num_samples
    num_steps = args.num_steps
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    stats_path = args.stats_path

    config = load_config(config_path)

    seq_len = config['model']['seq_len']
    checkpoint_name = config['model']['checkpoint_name']
    is_ema = config['train']['is_ema']

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

    label_mapping = {"maestro": 0,
                     "asap": 1}

    encdec = EncoderDecoder(device=device)

    os.makedirs(output_dir, exist_ok=True)

    gt_dir = os.path.join(output_dir, "groundtruth")
    rec_dir = os.path.join(output_dir, "reconstruction")
    deadpan_dir = os.path.join(output_dir, "deadpan")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)

    metadata = load_json(metadata_path)
    aligned_dict_lst = prepare_filepaths_from_aligned_metadata(metadata,
                                                               split='test')
    val_dict_lst = []
    for aligned_dict in aligned_dict_lst:
        emb_dict = {}
        emb_dict['filepath'] = aligned_dict['deadpan_path']
        emb_dict['source'] = 'asap'
        val_dict_lst.append(emb_dict)

    torch.manual_seed(42)
    val_dataset = EmbeddingDataset(emb_dict_lst=val_dict_lst,
                                   is_cache=True,
                                   seq_len=128,
                                   is_from_start=True)
    for x in val_dataset:
        pass

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False)

    label_source = label_mapping["asap"]
    y_source = torch.tensor([label_source] * num_samples, device=device)

    label_target = label_mapping["maestro"]
    y_target = torch.tensor([label_target] * num_samples, device=device)

    cfg_bridge = CFGBridge(model=sit,
                           interpolant=interpolant)
    cfg_source = 1
    cfg_target = 2

    for score_id, x1 in tqdm(enumerate(val_loader)):
        x1 = x1.to(device)
        x1 = torch.cat(num_samples * [x1])
        x1_rec = cfg_bridge.translate_sample(x=x1,
                                             y_source=y_source,
                                             y_target=y_target,
                                             cfg_source=cfg_source,
                                             cfg_target=cfg_target,
                                             num_steps=num_steps)

        out_score_dir = os.path.join(rec_dir, f"score_{score_id}")
        os.makedirs(out_score_dir, exist_ok=True)
        for i in range(len(x1_rec)):
            audio = encdec.decode(x1_rec[i].T.to(device))
            audio = audio.cpu().numpy()
            output_path = os.path.join(out_score_dir, f"rec_{i}.mp3")
            sf.write(output_path, audio.T, 44100)

        # Export deadpan audio
        out_score_dir = os.path.join(deadpan_dir, f"score_{score_id}")
        os.makedirs(out_score_dir, exist_ok=True)
        audio = encdec.decode(x1[0].T.to(device))
        audio = audio.cpu().numpy()
        output_path = os.path.join(out_score_dir, "deadpan.mp3")
        sf.write(output_path, audio.T, 44100)

    # Save expressive groundtruth samples
    val_dict_lst = []
    for aligned_dict in aligned_dict_lst:
        emb_dict = {}
        emb_dict['filepath'] = aligned_dict['expressive_path']
        emb_dict['source'] = 'maestro'
        val_dict_lst.append(emb_dict)

    torch.manual_seed(42)
    val_dataset = EmbeddingDataset(emb_dict_lst=val_dict_lst,
                                   is_cache=True,
                                   seq_len=seq_len,
                                   is_from_start=True)
    for x in val_dataset:
        pass

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False)

    for score_id, x1 in tqdm(enumerate(val_loader)):
        x1 = x1.to(device)
        out_score_dir = os.path.join(gt_dir, f"score_{score_id}")
        os.makedirs(out_score_dir, exist_ok=True)
        audio = encdec.decode(x1[0].T.to(device))
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
    parser.add_argument('--checkpoint_path', type=str, required=True, default=None, help='path to checkpoint')
    parser.add_argument('--stats_path', type=str, required=True, help='path to dataset embedding statistics')
    
    args = parser.parse_args()

    main(args)
