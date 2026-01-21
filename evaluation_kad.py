import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Union, Dict, List
import os

from hypy_utils.logging_utils import setup_logger

from kadtk.model_loader import get_all_models

from warping.utils import load_json
from warping.metrics.emb_loader import cache_embedding_files
from warping.metrics.fad import FrechetAudioDistance
from warping.metrics.kad import KernelAudioDistance


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

    parser = ArgumentParser()
    # Positional arguments
    parser.add_argument('model', type=str, choices=list(models.keys()),
                        help="The embedding model to use")
    parser.add_argument('baseline', type=str, help="The baseline dataset")
    parser.add_argument('eval', type=str, help="The dataset to evaluate against")
    parser.add_argument('metadata_path', type=str, help='path to eval metadata')

    # Alternative metric selection
    parser.add_argument('--fad', action='store_true',
                        help="Calculate Frechet Audio Distance (FAD). Default is Kernel Audio Distance (KAD).")

    # Optional parameters
    parser.add_argument('-b', '--bandwidth', type=float, default=None,
                        help="Set bandwidth for KAD. Adaptive bandwidth is used if not provided.")
    parser.add_argument('-w', '--workers', type=int, default=8,
                        help="Number of parallel workers for embedding calculation.")
    parser.add_argument('-s', '--sox-path', type=str, default='/usr/bin/sox',
                        help="Path to the sox executable.")
    parser.add_argument('--inf', action='store_true', help="Use FAD-inf extrapolation")
    parser.add_argument('--indiv', action='store_true', help="Calculate FAD for individual songs and store the results in the given file")
    parser.add_argument('--audio-len', type=int, default=None,
                        help="Length of audio clips in seconds.")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use for score calculation.")
    parser.add_argument('--csv', type=str, nargs='?',
                        help="Optional CSV file to append results. If omitted, results will be printed.")
    args = parser.parse_args()

    baseline = args.baseline
    eval = args.eval
    metadata_path = args.metadata_path
    metadata = load_json(metadata_path)
    eval = prepare_filepaths_from_metadata(metadata,
                                           audio_root_dir=eval,
                                           split='validation')
    baseline = list(Path(baseline).glob('*.*'))
    
    # Set up logger
    logger = setup_logger()

    # Load model
    if args.audio_len is not None:
        models = {m.name: m for m in get_all_models(audio_len=args.audio_len)}
    model = models[args.model]
    device = args.device

    # 1. Calculate embedding files for each dataset
    for d in [baseline, eval]:
        cache_embedding_files(d, model,
                              workers=args.workers)

    # 2. Calculate the chosen metric
    if args.fad:
        metric = FrechetAudioDistance(model, device, audio_load_worker=args.workers, logger=logger)
    else:
        metric = KernelAudioDistance(model, device, bandwidth=args.bandwidth, audio_load_worker=args.workers, logger=logger)
    
    if args.inf:
        score = metric.score_inf(baseline, list(Path(eval).glob('*.*')))
        score, inf_r2 = score.score, score.r2
    elif args.indiv:
        csv = metric.score_individual(baseline, eval, args.csv)
        logger.info(f"Individual FAD scores saved to {csv}")
        exit(0)
    else:
        score = metric.score(baseline, eval)
        inf_r2 = None

    # 3. Save or display results
    metric_name = 'FAD' if args.fad else 'KAD'
    logger.info(f"{metric_name} computed.")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not csv_path.is_file():
            csv_path.write_text('model,baseline,eval,score,time\n')
        with open(csv_path, 'a') as f:
            f.write(f'{model.name},{baseline},{eval},{score},{time.time()}\n')
        logger.info(f"{metric_name} score appended to {csv_path}")
    else:
        print(f"Score: {score}")

if __name__ == "__main__":
    main()
