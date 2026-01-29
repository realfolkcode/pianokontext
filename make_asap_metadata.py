"""
Creates metadata for ASAP dataset using splits from MAESTRO.
"""

import argparse
import os
import json
import pandas as pd

from warping.utils import load_json


def main(args):
    midi_metadata_path = args.midi_metadata
    maestro_metadata_path = args.maestro_metadata
    output_path = args.output_path

    metadata = pd.read_csv(midi_metadata_path)
    metadata = metadata.dropna(subset=["maestro_audio_performance"])

    maestro_metadata = load_json(maestro_metadata_path)
    maestro_metadata = pd.DataFrame.from_dict(maestro_metadata)

    metadata["maestro_midi_performance"] = metadata['maestro_midi_performance'].apply(lambda x: x.replace("{maestro}/", ""))
    metadata["maestro_audio_performance"] = metadata['maestro_audio_performance'].apply(lambda x: x.replace("{maestro}/", ""))

    maestro_metadata["maestro_audio_performance"] = maestro_metadata["audio_filename"]

    score_splits = pd.merge(metadata,
                            maestro_metadata[["split", "maestro_audio_performance"]],
                            how='left',
                            on="maestro_audio_performance")
    score_splits = score_splits.groupby('midi_score')["split"].apply(set).to_dict()

    # Make sure that all performances of the same score have the same split
    for split in score_splits.values():
        assert len(split) == 1

    for score, split in score_splits.items():
        score_splits[score] = list(split)[0]

    audio_metadata = {'audio_filename': {},
                      'split': {}}

    for i, midi_path in enumerate(score_splits.keys()):
        audio_path, _ = os.path.splitext(midi_path)
        audio_path = f"{audio_path}.wav"
        audio_metadata['audio_filename'][str(i)] = audio_path
        audio_metadata['split'][str(i)] = score_splits[midi_path]
    
    with open(output_path, "w", encoding="utf8") as f:
        f.write(json.dumps(audio_metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--midi_metadata', type=str, required=True, help='path to midi dataset metadata')
    parser.add_argument('--maestro_metadata', type=str, required=True, help='path to maestro metadata')
    parser.add_argument('--output_path', type=str, required=True, help='path to audio metadata')
    
    args = parser.parse_args()

    main(args)
