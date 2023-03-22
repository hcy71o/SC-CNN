# SC-CNN : Effective Speaker Conditioning Method for Zero-Shot Multi-Speaker Text-to-Speech Systems
Thanks to [StyleSpeech](https://arxiv.org/abs/2106.03153) and [VITS](https://arxiv.org/abs/2106.06103), we built up our codes based on [Link](https://github.com/KevinMIN95/StyleSpeech) and [Link](https://github.com/jaywalnut310/vits)

0. [VCTK]((https://paperswithcode.com/dataset/vctk)) dataset is used.
1. Sampling rate is set to 22050Hz.
2. This is the implementation of ```SC-TransferTTS```

## Materials
- [Demo page](https://hcy71o.github.io/SC-CNN-demo/)




## Prerequisites
0. Clone this repository.
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
0. Download datasets
    1. Download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY3`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace

## Training Exmaple
```sh
python train.py -c configs/vctk_base.json -m vctk_base
```

## Inference Example
See [inference.ipynb](inference.ipynb)
