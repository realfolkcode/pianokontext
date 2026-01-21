import os
import argparse
import pandas as pd
import subprocess
from tqdm import tqdm


def main(args):
    midi_root_dir = args.midi_dir
    metadata_path = args.midi_metadata
    audio_root_dir = args.audio_dir
    soundfont_path = args.soundfont_path

    metadata = pd.read_csv(metadata_path)
    midi_path_lst = list(metadata['midi_score'].to_dict().values())
    midi_path_lst = list(set(midi_path_lst))

    for midi_path in tqdm(midi_path_lst):
        midi_filename = os.path.basename(midi_path)
        sample_name, _ = os.path.splitext(midi_filename)
        midi_rel_dir = os.path.dirname(midi_path)
        midi_path = os.path.join(midi_root_dir, midi_path)

        audio_dir = os.path.join(audio_root_dir, midi_rel_dir)
        os.makedirs(audio_dir, exist_ok=True)

        audio_filename = f"{sample_name}.wav"
        audio_path = os.path.join(audio_dir, audio_filename)

        subprocess.run(["fluidsynth", "-ni", soundfont_path,
                        midi_path, "-F", audio_path, "-r", "44100"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--midi_dir', type=str, required=True, help='path to midi dataset directory')
    parser.add_argument('--midi_metadata', type=str, required=True, help='path to midi dataset metadata')
    parser.add_argument('--audio_dir', type=str, required=True, help='path to audio output directory')
    parser.add_argument('--soundfont_path', type=str, required=True, help='path to sf2 soundfont')
    
    args = parser.parse_args()

    main(args)
