# PianoKontext: Expressive Performance Rendering from Deadpan Context

## Installation

This repository requires Python 3.10 or greater.

```
# Run in your environment
pip install -r requirements.txt 
```

For training, you might need to install additional packages.

## Inference

The Colab notebook with an example of inference will be added soon

## Data preparation and training

1. Download the [MAESTRO](https://magenta.withgoogle.com/datasets/maestro) and [ASAP](https://github.com/fosfrancesco/asap-dataset) datasets.
2. Install [fluidsynth](https://www.fluidsynth.org/) and a piano soundfont. Synthesize ASAP from MIDI to audio using the `synthesize_asap.py` script.
3. Encode MAESTRO and ASAP with Music2Latent using the `encode_audio.py` script.
4. Align the embeddings using the `run_align_embeddings.py` script. It will produce the alignment files between ASAP and MAESTRO and a new metadata.
5. Calculate the embedding statistics. To this end, combine the rows from ASAP and MAESTRO metadata files. Run the `save_data_stats.py` script. It will produce the embedding statistics for a joint deadpan-expressive dataset.
6. Run the `run_flux_training.py` script to train a PianoKontext model.

## Acknowledgements
- [Music2Latent](https://github.com/SonyCSLParis/music2latent) for Music2Latent and pretrained checkpoints
- [FLUX Kontext](https://github.com/black-forest-labs/flux/tree/main/src/flux) for implementation details
- [KAD toolkit](https://github.com/YoonjinXD/kadtk/tree/main/kadtk) for evaluation metrics implementation
- [DTAIDistance](https://github.com/wannesm/dtaidistance) for fast DTW implementation
