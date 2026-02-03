import os
import json
import yaml
from typing import Dict, List, Union


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
                                    emb_root_dir: str,
                                    split: str | None = None,
                                    dataset_name: str | None = None) -> List[Dict]:
    """Prepares a list of embedding filepaths from metadata.

    Args:
        metadata: A dictionary with keywords `audio_filename` and `split`.
        emb_root_dir: The root directory of embeddings.
        split: Split to use. If None, takes all splits.
        dataset_name: The name of a dataset.
    
    Returns:
        A list of dictionaries with embedding filepaths and metadata.
    """
    emb_dict_lst = []

    audio_path_lst = list(metadata['audio_filename'].values())
    audio_split_lst = list(metadata['split'].values())

    for audio_path, audio_split in zip(audio_path_lst, audio_split_lst):
        if audio_split != split and split is not None:
            continue
        
        audio_filename = os.path.basename(audio_path)
        sample_name, _ = os.path.splitext(audio_filename)
        audio_rel_dir = os.path.dirname(audio_path)

        emb_dir = os.path.join(emb_root_dir, audio_rel_dir)
        emb_filename = f"{sample_name}.pt"
        emb_path = os.path.join(emb_dir, emb_filename)

        emb_dict = {"filepath": emb_path,
                    "dataset_name": dataset_name}

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
