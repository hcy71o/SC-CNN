import torch
import numpy as np
import os
import argparse
import librosa
import re
import json
import glob
from string import punctuation
from g2p_en import G2p

from models.StyleSpeech import StyleSpeech
from text import text_to_sequence
import audio as Audio
import utils
import random


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


def preprocess_audio(audio_file, _stft, db):
    wav, sample_rate = librosa.load(audio_file, sr=None)
    if sample_rate != 16000:
        wav = librosa.resample(wav, sample_rate, 16000)
    '''
    It is shown that the trimming should be carefully designed,
    because the output audio quality highly depends on the input ref audio.
    Recommend to listen trimmed ref audio before inference.
    '''
    wav, section = librosa.effects.trim(wav, top_db=db)
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _stft)
    return torch.from_numpy(mel_spectrogram).to(device=device)


def get_StyleSpeech(config, checkpoint_path):
    model = StyleSpeech(config).to(device=device)
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.eval()
    return model


def synthesize(args, audio, text, model, _stft):   
    # preprocess audio and text

    ref_mel = preprocess_audio(audio, _stft, args.top_db).transpose(0,1).unsqueeze(0)
    
    save_path = args.save_path + '_' + str(args.top_db) + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Extract style vector
    style_vector = model.get_style_vector(ref_mel)
    mel_ref_ = ref_mel.cpu().squeeze().transpose(0, 1).detach()
    np.save(save_path + '{}_ref.npy'.format(audio[-12:-4]), np.array(mel_ref_.unsqueeze(0)))
    
    if isinstance(text, list):
    # Forward
        for txt in text:
            src = preprocess_english(txt, args.lexicon_path).unsqueeze(0)
            src_len = torch.from_numpy(np.array([src.shape[1]])).to(device=device)
            mel_output = model.inference(style_vector, src, src_len)[0]
            mel_ = mel_output.cpu().squeeze().transpose(0, 1).detach()
            name = audio[-12:-4] + '_' + txt[:10].replace(' ', '_')
            np.save(save_path + '{}.npy'.format(name), np.array(mel_.unsqueeze(0)))
            
    else:
        src = preprocess_english(text, args.lexicon_path).unsqueeze(0)
        src_len = torch.from_numpy(np.array([src.shape[1]])).to(device=device)
        mel_output = model.inference(style_vector, src, src_len)[0]
        mel_ = mel_output.cpu().squeeze().transpose(0, 1).detach()
        name = audio[-12:-4] + '_' + text[:10].replace(' ', '_')
        np.save(save_path + '{}.npy'.format(name), np.array(mel_.unsqueeze(0)))

    # plotting
    # utils.plot_data([mel_ref_.numpy(), mel_.numpy()], 
    #     ['Ref Spectrogram', 'Synthesized Spectrogram'], filename=os.path.join(save_path, 'plot.png'))
    
    print('Generate done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default = 'exp_ch8_ker7/ckpt/checkpoint_800000.pth.tar', 
        help="Path to the pretrained model")
    # parser.add_argument('--config', default='configs/config.json')
    parser.add_argument('--config', default='exp_ch8_ker7/config.json')
    parser.add_argument("--save_path", type=str, default='Final_SMOS')
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--ref_audio", type=str, default='/home/hcy71/VCTK-Corpus',
        help="path to an reference speech audio sample")
    parser.add_argument("--top_db", type=int, default=12),
    # parser.add_argument("--ref_spk", type=str, default = None)
    parser.add_argument("--text", type=str, default='In being comparatively modern.',
        help="raw text to synthesize")
    parser.add_argument("--lexicon_path", type=str, default='lexicon/librispeech-lexicon.txt')
    args = parser.parse_args()
    
    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    config = utils.AttrDict(json_config)

    # Get model
    model = get_StyleSpeech(config, args.checkpoint_path)
    print('model is prepared')

    _stft = Audio.stft.TacotronSTFT(
                config.filter_length,
                config.hop_length,
                config.win_length,
                config.n_mel_channels,
                config.sampling_rate,
                config.mel_fmin,
                config.mel_fmax)
    #TODO 이건 SMOS를 위한 거임
    total_audios = glob.glob(args.ref_audio+'/wav48/*/*.wav')
    random.seed(71)
    selected_audios = random.sample(total_audios, args.total)
    selected_audios_2 = random.sample(total_audios, 80)
    num_list_2 = [7,20,24,46,10,77]
    for i, audio in enumerate(selected_audios_2):
        if i in num_list_2:
            print(i, audio)
            txt_path = audio.replace('.wav','.txt')
            txt_path = txt_path.replace('wav48','txt')
            # try:
            txt = open(txt_path, 'r').read()
            print(audio[-12:-4], txt)
            synthesize(args, audio, txt, model, _stft)
    
    print(len(selected_audios))
    trimmed_m = [2,7,21,27,30,32,37,40,45,59,63,68,76,77,93,95,97]
    trimmed_f = [0,6,8,13,16,20,26,36,42,46,62,64,70,75,80,81,84,87,88,89,92,98]
    random.shuffle(trimmed_m)
    random.shuffle(trimmed_f)
    samples = trimmed_m[:12] + trimmed_f[:16]
    samples = samples + random.sample([21,27,30,40],3) + random.sample([26,46,84],3)
    removal = [16,32,37, 64,68,77,87,92] + [21,26,40,97] + [8,26,36,42, 30, 63,80]
    samples = samples + [7,93, 39, 54, 59, 78]
    samples = trimmed_f + trimmed_m
    
    for i, audio in enumerate(selected_audios):
        if i in samples: #and i not in removal:
            print(i, audio)
            txt_path = audio.replace('.wav','.txt')
            txt_path = txt_path.replace('wav48','txt')
            # try:
            txt = open(txt_path, 'r').read()
            print(i, '/', args.total, '--', audio[-12:-4], txt)
            synthesize(args, audio, txt, model, _stft)
    
    
    # Synthesize