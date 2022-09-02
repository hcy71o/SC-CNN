# StyleSpeech : Multi-Speaker Adaptive Text-to-Speech Generation
Unofficial PyTorch Implementation of [paper](https://arxiv.org/abs/2106.03153).
Most of codes are based on [Link](https://github.com/KevinMIN95/StyleSpeech)

0. [LibriTTS]((https://research.google/tools/datasets/libri-tts/)) dataset (train-clean-100 and train-clean-360) is used.
1. Sampling rate is set to 22050Hz (default).
## Prerequisites
- Clone this repository.
- Install python requirements. Please refer [requirements.txt](requirements.txt)

## Preparing
0. Run 
```
python prepare_align.py --data_path [LibriTTS DATAPATH]
```
for some preparations. (You can change the sampling rate by adding --resample_rate [SR])
1. [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences. 
1-1. Download MFA following the command in the website.
1-2. Run the below codes
```
$ conda activate aligner
$ mfa model download acoustic english_mfa
$ mfa align ......LibriTTS/wav22 lexicon.txt english_us_arpa .........LibriTTS/Textgrid
```
2. Run 
```
python preprocess.py
```
(Check input&output data paths)

## Training
```
python train.py --data_path [Preprocessed LibriTTS DATAPATH]
```

## Inference
0. Mel generation
```
python synthesize.py --checkpoint_path [CKPT PATH] --ref_audio [REF AUDIO PATH]
```
1. Waveform generation (Use hifi-gan)
```
cd hifi-gan
python inference_e2e.py --checkpoint_file [VOCODER CKPT PATH]
```