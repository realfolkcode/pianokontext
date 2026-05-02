import os
import argparse
import pandas as pd
import torch
import json
from tqdm import tqdm

from warping.utils import load_json, construct_filepath
from warping.alignment import cut_embedding, calculate_dtw_path


def main(args):
    paired_metadata_path = args.paired_metadata_path
    output_dir = args.output_dir
    output_metadata_path = args.output_metadata_path

    paired_metadata = load_json(paired_metadata_path)
    paired_metadata = pd.DataFrame.from_dict(paired_metadata)

    paired_metadata["alignment_dir"] = output_dir

    os.makedirs(output_dir, exist_ok=True)

    for i, row in tqdm(paired_metadata.iterrows()):
        expressive_emb_dir = row["expressive_emb_dir"]
        expressive_filepath = row["expressive_filename"]
        emb_dir, emb_filename = construct_filepath(new_root_dir=expressive_emb_dir,
                                                   audio_filename=expressive_filepath,
                                                   ext="pt")
        emb_path = os.path.join(emb_dir, emb_filename)
        expressive_emb = torch.load(emb_path)
        expressive_emb = expressive_emb.squeeze().T
        expressive_start = row["start"]
        expressive_end = row["end"]
        expressive_emb = cut_embedding(expressive_emb,
                                       start_s=expressive_start,
                                       end_s=expressive_end)

        deadpan_emb_dir = row["deadpan_emb_dir"]
        deadpan_filepath = row["deadpan_filename"]
        emb_dir, emb_filename = construct_filepath(new_root_dir=deadpan_emb_dir,
                                                   audio_filename=deadpan_filepath,
                                                   ext="pt")
        emb_path = os.path.join(emb_dir, emb_filename)
        deadpan_emb = torch.load(emb_path)
        deadpan_emb = deadpan_emb.squeeze().T

        dtw_path = calculate_dtw_path(deadpan_emb=deadpan_emb,
                                      expressive_emb=expressive_emb)
        alignment_filename = f"alignment_{i}.pt"
        alignment_filepath = os.path.join(output_dir, alignment_filename)
        torch.save(dtw_path, alignment_filepath)

    with open(output_metadata_path, "w", encoding="utf8") as f:
        f.write(json.dumps(paired_metadata.to_dict(), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired_metadata_path', type=str, required=True, help='path to paired dataset metadata')
    parser.add_argument('--output_dir', type=str, required=True, help='output dir to store alignments')
    parser.add_argument('--output_metadata_path', type=str, required=True, help='output path to new metadata')

    args = parser.parse_args()

    main(args)
