import time
from argparse import ArgumentParser
from typing import Dict, List
import os
from pathlib import Path

from fadtk.model_loader import *
from fadtk.fad_batch import cache_embedding_files

from warping.utils import load_json
from warping.fad import FrechetAudioDistance, log


def prepare_filepaths_from_metadata(metadata: Dict,
                                    audio_root_dir: str,
                                    split: str | None = None) -> List[str]:
    """Prepares a list of audio filepaths from MAESTRO metadata.

    Args:
        metadata: A dictionary with keywords `audio_filename` and `split`.
        audio_root_dir: The root directory of audio.
        split: Split to use. If None, takes all splits.
    
    Returns:
        A list of audio filepaths from MAESTRO metadata.
    """
    path_lst = []

    audio_path_lst = list(metadata['audio_filename'].values())
    audio_split_lst = list(metadata['split'].values())

    for audio_path, audio_split in zip(audio_path_lst, audio_split_lst):
        if audio_split != split and split is not None:
            continue
        path_lst.append(os.path.join(audio_root_dir,
                                     audio_path))
    
    return path_lst


def main():
    """
    Launcher for running FAD on two directories using a model.
    """
    models = {m.name: m for m in get_all_models()}

    agupa = ArgumentParser()
    # Two positional arguments: model and two directories
    agupa.add_argument('model', type=str, choices=list(models.keys()), help="The embedding model to use")
    agupa.add_argument('baseline', type=str, help="The baseline dataset")
    agupa.add_argument('eval', type=str, help="The directory to evaluate against")
    agupa.add_argument('metadata_path', type=str, help='path to eval metadata')
    agupa.add_argument('csv', type=str, nargs='?',
                       help="The CSV file to append results to. "
                            "If this argument is not supplied, single-value results will be printed to stdout, "
                            "and for --indiv, the results will be saved to 'fad-individual-results.csv'")

    # Add optional arguments
    agupa.add_argument('-w', '--workers', type=int, default=8)
    agupa.add_argument('-s', '--sox-path', type=str, default='/usr/bin/sox')

    args = agupa.parse_args()
    model = models[args.model]

    baseline = args.baseline
    eval = args.eval
    metadata_path = args.metadata_path
    metadata = load_json(metadata_path)
    eval = prepare_filepaths_from_metadata(metadata,
                                           audio_root_dir=eval,
                                           split='validation')
    baseline = list(Path(baseline).glob('*.*'))

    # 1. Calculate embedding files for each dataset
    for d in [baseline, eval]:
        cache_embedding_files(d, model, workers=args.workers)

    # 2. Calculate FAD
    fad = FrechetAudioDistance(model, audio_load_worker=args.workers, load_model=False)
    score = fad.score(baseline, eval)
    inf_r2 = None

    # 3. Print results    
    log.info("FAD computed.")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        if not Path(args.csv).is_file():
            Path(args.csv).write_text('model,baseline,eval,score,inf_r2,time\n')
        with open(args.csv, 'a') as f:
            f.write(f'{model.name},{baseline},{eval},{score},{inf_r2},{time.time()}\n')
        log.info(f"FAD score appended to {args.csv}")

    log.info(f"The FAD {model.name} score between {baseline} and {eval} is: {score}")


if __name__ == "__main__":
    main()
