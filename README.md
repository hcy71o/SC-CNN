# SC-CNN : An Effective Style Conditioning Method for Zero-Shot Text-to-Speech Systems
Thanks to [StyleSpeech](https://arxiv.org/abs/2106.03153), we built up our codes based on [Link](https://github.com/KevinMIN95/StyleSpeech)

0. [LibriTTS]((https://research.google/tools/datasets/libri-tts/)) dataset (train-clean-100 and train-clean-360) is used.
1. You can select sampling rate for both 22050Hz and 16000Hz.

## Materials
- [Demo page](https://hcy71o.github.io/SC-CNN-demo/)
<!-- - [TODO][Pretrained Model] -->
<!-- - [TODO] Add pretrained HiFi-GAN and its inference code -->

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
$ mfa align ......LibriTTS/wav16 lexicon.txt english_us_arpa .........LibriTTS/Textgrid
or
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
You can change hyperparameters of SC-CNN (kernel_size, channels), sampling rate (16 or 22.05kHz) or other model configurations in configs/config.json

## Inference
0. Mel generation
```
python synthesize.py --checkpoint_path [CKPT PATH] --ref_audio [REF AUDIO PATH] --text [INPUT TEXT]
```
