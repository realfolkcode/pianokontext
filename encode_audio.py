import os
import argparse
import torch
import librosa
from tqdm import tqdm

from warping.utils import load_json
from warping.backbone import EncoderDecoder
from codicodec import EncoderDecoder as CodicodecEncoderDecoder


def main(args):
    audio_root_dir = args.audio_dir
    metadata_path = args.audio_metadata
    emb_root_dir = args.emb_dir
    backbone = args.backbone

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert backbone in ["music2latent", "codicodec"]
    if backbone in "music2latent":
        encdec = EncoderDecoder(device=device)
    else:
        encdec = CodicodecEncoderDecoder(device=device)

    os.makedirs(emb_root_dir, exist_ok=True)

    metadata = load_json(metadata_path)

    for audio_path in tqdm(metadata["audio_filename"].values()):
        audio_filename = os.path.basename(audio_path)
        sample_name, _ = os.path.splitext(audio_filename)
        audio_rel_dir = os.path.dirname(audio_path)
        audio_path = os.path.join(audio_root_dir, audio_path)

        emb_dir = os.path.join(emb_root_dir, audio_rel_dir)
        os.makedirs(emb_dir, exist_ok=True)

        emb_filename = f"{sample_name}.pt"
        emb_path = os.path.join(emb_dir, emb_filename)

        audio, sr = librosa.load(audio_path, sr=44100)
        emb = encdec.encode(audio).cpu()

        torch.save(emb, emb_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True, help='path to audio dataset directory')
    parser.add_argument('--audio_metadata', type=str, required=True, help='path to audio dataset metadata')
    parser.add_argument('--emb_dir', type=str, required=True, help='output path to embedding dataset directory')
    parser.add_argument('--backbone', type=str, required=True, help='music2latent or codicodec model')

    args = parser.parse_args()

    main(args)
