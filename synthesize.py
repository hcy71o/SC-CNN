import torch
import numpy as np
import os
import argparse
import librosa
import re
import json
from string import punctuation
from g2p_en import G2p

from models.SCCNN import SCCNN
from text import text_to_sequence
import audio as Audio
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, lexicon_path):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(lexicon_path)

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(text_to_sequence(phones, ['english_cleaners']))

    return torch.from_numpy(sequence).to(device=device)


def preprocess_audio(args, _stft):
    wav, sample_rate = librosa.load(args.ref_audio, sr=None)
    if sample_rate != args.sampling_rate:
        wav = librosa.resample(wav, sample_rate, args.sampling_rate)
    '''
    Experimentally, trimming the reference audio enhance the generation quality.
    For high quality, we recommend to listen trimmed ref audio before inference.
    We set default top_db=20 for VCTK dataset.
    '''
    wav, section = librosa.effects.trim(wav, top_db=20)
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _stft)
    return torch.from_numpy(mel_spectrogram).to(device=device)


def get_SCCNN(config, checkpoint_path):
    model = SCCNN(config).to(device=device)
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.eval()
    return model


def synthesize(args, text, model, _stft):   
    # preprocess audio and text

    ref_mel = preprocess_audio(args, _stft).transpose(0,1).unsqueeze(0)
    
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Extract style vector
    style_vector = model.get_style_vector(ref_mel)
    mel_ref_ = ref_mel.cpu().squeeze().transpose(0, 1).detach()
    np.save(save_path + 'ref_{}.npy'.format(args.ref_audio[-12:-4]), np.array(mel_ref_.unsqueeze(0)))
    
    if isinstance(text, list):
    # Forward
        for txt in text:
            src = preprocess_english(txt, args.lexicon_path).unsqueeze(0)
            src_len = torch.from_numpy(np.array([src.shape[1]])).to(device=device)
            mel_output = model.inference(style_vector, src, src_len)[0]
            mel_ = mel_output.cpu().squeeze().transpose(0, 1).detach()
            name = args.ref_audio[-12:-4] + '_' + txt[:10].replace(' ', '_')
            np.save(save_path + '{}.npy'.format(name), np.array(mel_.unsqueeze(0)))
            
    else:
        src = preprocess_english(text, args.lexicon_path).unsqueeze(0)
        src_len = torch.from_numpy(np.array([src.shape[1]])).to(device=device)
        mel_output = model.inference(style_vector, src, src_len)[0]
        mel_ = mel_output.cpu().squeeze().transpose(0, 1).detach()
        name = args.ref_audio[-12:-4] + '_' + text[:10].replace(' ', '_')
        np.save(save_path + '{}.npy'.format(name), np.array(mel_.unsqueeze(0)))
    
    print('Generate done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, 
        help="Path to the pretrained model")
    # parser.add_argument('--config', default='configs/config.json')
    parser.add_argument('--config', default='exp_ch8_ker7/config.json')
    parser.add_argument("--save_path", type=str, default='results/')
    parser.add_argument("--ref_audio", type=str, required=True,
        help="path to an reference speech audio sample")
    parser.add_argument("--ref_spk", type=str, default = None)
    parser.add_argument("--text", type=str, default='In being comparatively modern.',
        help="raw text to synthesize")
    parser.add_argument("--lexicon_path", type=str, default='lexicon/librispeech-lexicon.txt')
    args = parser.parse_args()
    
    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    config = utils.AttrDict(json_config)

    # Get model
    model = get_SCCNN(config, args.checkpoint_path)
    print('model is prepared')

    _stft = Audio.stft.TacotronSTFT(
                config.filter_length,
                config.hop_length,
                config.win_length,
                config.n_mel_channels,
                config.sampling_rate,
                config.mel_fmin,
                config.mel_fmax)

    # Synthesize
    args.text = [
        'Please call Stella.',
        'Ask her to bring these things with her from the store.',
        'Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.',
        'We also need a small plastic snake and a big toy frog for the kids.',
        'When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.',
        # 'People put on their coats, is it cold out there?'
    ]
    
    synthesize(args, args.text, model, _stft)
