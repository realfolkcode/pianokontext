import os
import json
import yaml
import pandas as pd
from typing import Dict, List, Union, Tuple


def load_json(json_path: str) -> Union[Dict, List[Dict]]:
    """Loads json.

    Args:
        json_path: The path to json.

    Returns:
        The content of json file.
    """
    with open(json_path, "r", encoding="utf8") as f:
        json_content = json.load(f)
    return json_content


def prepare_filepaths_from_metadata(metadata: Dict,
                                    split: str | None = None) -> List[Dict]:
    """Prepares a list of embedding filepaths from metadata.

    Args:
        metadata: A dictionary with keywords `audio_filename` and `split`.
        split: Split to use. If None, takes all splits.
    
    Returns:
        A list of dictionaries with embedding filepaths and metadata.
    """
    emb_dict_lst = [] 
    
    metadata = pd.DataFrame.from_dict(metadata)
    for i, row in metadata.iterrows():
        audio_path = row['audio_filename']
        emb_root_dir = row['embeddings_dir']
        audio_split = row['split']
        source = row['source']

        if audio_split != split and split is not None:
            continue

        audio_filename = os.path.basename(audio_path)
        sample_name, _ = os.path.splitext(audio_filename)
        audio_rel_dir = os.path.dirname(audio_path)

        emb_dir = os.path.join(emb_root_dir, audio_rel_dir)
        emb_filename = f"{sample_name}.pt"
        emb_path = os.path.join(emb_dir, emb_filename)

        emb_dict = {"filepath": emb_path,
                    "source": source}

        emb_dict_lst.append(emb_dict)
    
    return emb_dict_lst


def construct_filepath(new_root_dir: str,
                       audio_filename: str,
                       ext: str) -> Tuple[str, str]:
    """
    Replaces the audio filename with the new filename given the root directory
    and a new extension.

    Args:
        new_root_dir: New root directory.
        audio_filename: Relative audio path.
        ext: New extension.
    
    Returns:
        New directory and a filepath.
    """
    new_dir = os.path.join(new_root_dir,
                           os.path.dirname(audio_filename))
    filename = os.path.basename(audio_filename)
    filename, _ = os.path.splitext(filename)
    filename = f"{filename}.{ext}"
    return new_dir, filename


def prepare_filepaths_from_aligned_metadata(
    metadata: Dict,
    split: str | None = None
) -> List[Dict]:
    """Prepares a list of embedding filepaths from aligned metadata.

    Args:
        metadata: A dictionary with metadata.
        split: Split to use. If None, takes all splits.
    
    Returns:
        A list of dictionaries with embedding filepaths and alignment.
    """
    emb_dict_lst = [] 
    
    metadata = pd.DataFrame.from_dict(metadata)
    for i, row in metadata.iterrows():
        audio_split = row["split"]
        if audio_split != split and split is not None:
            continue

        # Process expressive metadata
        expressive_emb_dir = row["expressive_emb_dir"]
        expressive_filepath = row["expressive_filename"]
        emb_dir, emb_filename = construct_filepath(new_root_dir=expressive_emb_dir,
                                                   audio_filename=expressive_filepath,
                                                   ext="pt")
        expressive_path = os.path.join(emb_dir, emb_filename)
        expressive_start = row["start"]
        expressive_end = row["end"]

        # Process deadpan metadata
        deadpan_emb_dir = row["deadpan_emb_dir"]
        deadpan_filepath = row["deadpan_filename"]
        emb_dir, emb_filename = construct_filepath(new_root_dir=deadpan_emb_dir,
                                                   audio_filename=deadpan_filepath,
                                                   ext="pt")
        deadpan_path = os.path.join(emb_dir, emb_filename)

        # Process alignment metadata
        alignment_dir = row["alignment_dir"]
        alignment_dir, alignment_filename = construct_filepath(new_root_dir=alignment_dir,
                                                               audio_filename=expressive_filepath,
                                                               ext='pt')
        alignment_filepath = os.path.join(alignment_dir, alignment_filename)

        # Gather dictionary
        emb_dict = {"expressive_path": expressive_path,
                    "deadpan_path": deadpan_path,
                    "alignment_filepath": alignment_filepath,
                    "expressive_start_s": expressive_start,
                    "expressive_end_s": expressive_end}
        emb_dict_lst.append(emb_dict)
    
    return emb_dict_lst


def load_config(config_path: str) -> Dict:
    """Loads a yaml config for setting experiment parameters.
    
    Args:
        config_path: The path to a yaml config.
    
    Returns:
        A dictionary with experiment parameters.
    """
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
