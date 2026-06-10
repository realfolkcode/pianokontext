# PianoKontext: Expressive Performance Rendering from Deadpan Context

Accepted at the ICML 2026 Workshop on Machine Learning for Audio

Link to the paper will be added soon

[![Demo page with audio examples](https://img.shields.io/badge/Website-Demo-2563eb)](https://realfolkcode.github.io/pianokontext_demo)

## What is PianoKontext?

PianoKontext is a proof-of-concept model for **variable-length** expressive rendering of classial piano music. Given a deadpan audio synthesized from a MIDI score, it generates various expressive audios with different timings and dynamics. 

Inspired by FLUX Kontext, PianoKontext is a flow matching model trained in the latent space of Music2Latent that enables contextual learning of score-performance dependencies solely through self-attention. Currently, it operates on segments up to 11 seconds.

**Try it on Google Colab!**

<a target="_blank" href="https://colab.research.google.com/drive/1cKoKdoRKZd89gvBsnhYdsxyKsR-bS9QP?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Installation

This repository requires Python 3.10 or greater.

```
# Run in your environment
pip install -r requirements.txt 
```

For training, you might need to install additional packages.

## Data preparation and training

More details will be added soon

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
