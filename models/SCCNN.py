import torch
import torch.nn as nn
import numpy as np
from text.symbols import symbols
import models.Constants as Constants
from models.Modules import Mish, LinearNorm, ConvNorm, Conv1dGLU, \
                    MultiHeadAttention, StyleAdaptiveLayerNorm, get_sinusoid_encoding_table
from models.VarianceAdaptor import VarianceAdaptor
from models.Loss import StyleSpeechLoss
from utils import get_mask_from_lengths


class SCCNN(nn.Module):
    ''' SCCNN '''
    def __init__(self, config):
        super(SCCNN, self).__init__()
        self.style_encoder = MelStyleEncoder(config)
        self.encoder = Encoder(config)
        self.variance_adaptor = VarianceAdaptor(config)
        self.decoder = Decoder(config)
        
    def parse_batch(self, batch):
        sid = torch.from_numpy(batch["sid"]).long().cuda()
        text = torch.from_numpy(batch["text"]).long().cuda()
        mel_target = torch.from_numpy(batch["mel_target"]).float().cuda()
        D = torch.from_numpy(batch["D"]).long().cuda()
        log_D = torch.from_numpy(batch["log_D"]).float().cuda()
        f0 = torch.from_numpy(batch["f0"]).float().cuda()
        energy = torch.from_numpy(batch["energy"]).float().cuda()
        src_len = torch.from_numpy(batch["src_len"]).long().cuda()
        mel_len = torch.from_numpy(batch["mel_len"]).long().cuda()
        max_src_len = np.max(batch["src_len"]).astype(np.int32)
        max_mel_len = np.max(batch["mel_len"]).astype(np.int32)
        return sid, text, mel_target, D, log_D, f0, energy, src_len, mel_len, max_src_len, max_mel_len

    def forward(self, src_seq, src_len, mel_target, mel_len=None, 
                    d_target=None, p_target=None, e_target=None, max_src_len=None, max_mel_len=None):
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None
        
        # Extract Style Vector
        style_vector = self.style_encoder(mel_target, mel_mask)
        # Encoding
        encoder_output, src_embedded, _ = self.encoder(src_seq, style_vector, src_mask)
        # Variance Adaptor
        acoustic_adaptor_output, d_prediction, p_prediction, e_prediction, mel_len, mel_mask = self.variance_adaptor(
                encoder_output, src_mask, mel_len, mel_mask, 
                        d_target, p_target, e_target, max_mel_len)
        # Deocoding
        mel_prediction, _ = self.decoder(acoustic_adaptor_output, style_vector, mel_mask)

        return mel_prediction, src_embedded, style_vector, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len

    def inference(self, style_vector, src_seq, src_len=None, max_src_len=None, return_attn=False, return_kernel=False):
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        
        # Encoding
        if return_kernel:
            kernel_params = self.encoder(src_seq, style_vector, src_mask, True)
            return kernel_params
        
        encoder_output, src_embedded, enc_slf_attn = self.encoder(src_seq, style_vector, src_mask)


        # Variance Adaptor
        acoustic_adaptor_output, d_prediction, p_prediction, e_prediction, \
                mel_len, mel_mask = self.variance_adaptor(encoder_output, src_mask)

        # Deocoding
        mel_output, dec_slf_attn = self.decoder(acoustic_adaptor_output, style_vector, mel_mask)

        if return_attn:
            return enc_slf_attn, dec_slf_attn

        return mel_output, src_embedded, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len

    def get_style_vector(self, mel_target, mel_len=None):
        mel_mask = get_mask_from_lengths(mel_len) if mel_len is not None else None
        style_vector = self.style_encoder(mel_target, mel_mask)

        return style_vector

    def get_criterion(self):
        return StyleSpeechLoss()


class Encoder(nn.Module):
    ''' Encoder '''
    def __init__(self, config, n_src_vocab=len(symbols)+1):
        super(Encoder, self).__init__()
        self.max_seq_len = config.max_seq_len
        self.n_layers = config.encoder_layer
        self.d_model = config.encoder_hidden
        self.n_head = config.encoder_head
        self.d_k = config.encoder_hidden // config.encoder_head
        self.d_v = config.encoder_hidden // config.encoder_head
        self.d_hid1 = config.enc_ffn_in_ch_size
        self.d_hid2 = config.enc_ffn_out_ch_size
        self.fft_conv1d_kernel_size = config.fft_conv1d_kernel_size
        self.d_out = config.decoder_hidden
        self.style_dim = config.style_vector_dim
        self.dropout = config.dropout

        self.src_word_emb = nn.Embedding(n_src_vocab, self.d_model, padding_idx=Constants.PAD)
        self.prenet = Prenet(self.d_model, self.d_model, self.dropout)

        n_position = self.max_seq_len + 1
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, self.d_model).unsqueeze(0), requires_grad = False)

        self.layer_stack = nn.ModuleList([SCFFTBlock(
            self.d_model, self.d_hid1, self.d_hid2, self.n_head, self.d_k, self.d_v, 
            self.style_dim, self.dropout) for _ in range(self.n_layers)])

        self.fc_out = nn.Linear(self.d_model, self.d_out)
        
        self.kernel_predictor = KernelPredictor(self.style_dim, self.n_layers, config.enc_ffn_style_conv1d_kernel_size, 
                                                config.enc_ffn_in_ch_size, config.enc_ffn_out_ch_size)

    def forward(self, src_seq, style_vector, mask, return_kernels=False):
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        
        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        # word embedding
        src_embedded = self.src_word_emb(src_seq)
        # prenet
        src_seq = self.prenet(src_embedded, mask)
        # position encoding
        if src_seq.shape[1] > self.max_seq_len:
            position_embedded = get_sinusoid_encoding_table(src_seq.shape[1], self.d_model)[:src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(src_seq.device)
        else:
            position_embedded = self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        enc_output = src_seq + position_embedded
        # fft blocks
        slf_attn = []
        
        kernel_params = self.kernel_predictor(style_vector)
        d_ws, d_gs, d_bs, p_ws, p_gs, p_bs = kernel_params
        if return_kernels:
            return kernel_params
        
        for i, enc_layer in enumerate(self.layer_stack):
            
            d_w, d_g, d_b, p_w, p_g, p_b = d_ws[:, i, :, :, :], d_gs[:, i, :], d_bs[:, i, :], \ 
                                            p_ws[:, i, :, :, :], p_gs[:, i, :], p_bs[:, i, :]
            
            enc_output, enc_slf_attn = enc_layer(
                enc_output, d_w, d_g, d_b, p_w, p_g, p_b,
                mask=mask, 
                slf_attn_mask=slf_attn_mask)
            slf_attn.append(enc_slf_attn)
        # last fc
        enc_output = self.fc_out(enc_output)
        return enc_output, src_embedded, slf_attn


class Decoder(nn.Module):
    """ Decoder """
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.max_seq_len = config.max_seq_len
        self.n_layers = config.decoder_layer
        self.d_model = config.decoder_hidden
        self.n_head = config.decoder_head
        self.d_k = config.decoder_hidden // config.decoder_head
        self.d_v = config.decoder_hidden // config.decoder_head
        self.d_inner = config.fft_conv1d_filter_size
        self.fft_conv1d_kernel_size = config.fft_conv1d_kernel_size
        self.d_out = config.n_mel_channels
        self.style_dim = config.style_vector_dim
        self.dropout = config.dropout

        self.prenet = nn.Sequential(
            nn.Linear(self.d_model, self.d_model//2),
            Mish(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model//2, self.d_model)
        )

        n_position = self.max_seq_len + 1
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, self.d_model).unsqueeze(0), requires_grad = False)

        self.layer_stack = nn.ModuleList([FFTBlock(
            self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, 
            self.fft_conv1d_kernel_size, self.style_dim, self.dropout) for _ in range(self.n_layers)])

        self.fc_out = nn.Linear(self.d_model, self.d_out)

    def forward(self, enc_seq, style_code, mask):
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]
        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        # prenet
        dec_embedded = self.prenet(enc_seq)
        # poistion encoding
        if enc_seq.shape[1] > self.max_seq_len:
            position_embedded = get_sinusoid_encoding_table(enc_seq.shape[1], self.d_model)[:enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(enc_seq.device)
        else:
            position_embedded = self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        dec_output = dec_embedded + position_embedded
        # fft blocks
        slf_attn = []
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, style_code,
                mask=mask,
                slf_attn_mask=slf_attn_mask)
            slf_attn.append(dec_slf_attn)
        # last fc
        dec_output = self.fc_out(dec_output)
        return dec_output, slf_attn


class FFTBlock(nn.Module):
    ''' FFT Block '''
    def __init__(self, d_model,d_inner,
                    n_head, d_k, d_v, fft_conv1d_kernel_size, style_dim, dropout):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.saln_0 = StyleAdaptiveLayerNorm(d_model, style_dim)

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel_size, dropout=dropout)
        self.saln_1 = StyleAdaptiveLayerNorm(d_model, style_dim)

    def forward(self, input, style_vector, mask=None, slf_attn_mask=None):
        # multi-head self attn
        slf_attn_output, slf_attn = self.slf_attn(input, mask=slf_attn_mask)
        slf_attn_output = self.saln_0(slf_attn_output, style_vector)
        if mask is not None:
            slf_attn_output = slf_attn_output.masked_fill(mask.unsqueeze(-1), 0)

        # position wise FF
        output = self.pos_ffn(slf_attn_output)
        output = self.saln_1(output, style_vector)
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)

        return output, slf_attn
    
class SCFFTBlock(nn.Module):
    ''' 
    Revised FFT Block 
    '''
    def __init__(self, d_model,d_hid1,d_hid2,
                    n_head, d_k, d_v, style_dim, dropout):
        super(SCFFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.ln_0 = nn.LayerNorm(d_model)

        self.pos_ffn = SCPositionwiseFeedForward(
            d_model, d_hid1, d_hid2, dropout=dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        

    def forward(self, input, d_w, d_g, d_b, p_w, p_g, p_b, mask=None, slf_attn_mask=None):
        # multi-head self attn
        slf_attn_output, slf_attn = self.slf_attn(input, mask=slf_attn_mask)
        slf_attn_output = self.ln_0(slf_attn_output)
        if mask is not None:
            slf_attn_output = slf_attn_output.masked_fill(mask.unsqueeze(-1), 0)

        # position wise FF
        output = self.pos_ffn(slf_attn_output, d_w, d_g, d_b, p_w, p_g, p_b)
        output = self.ln_1(output)
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)

        return output, slf_attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, fft_conv1d_kernel_size, dropout=0.1):
        super().__init__()
        self.w_1 = ConvNorm(d_in, d_hid, kernel_size=fft_conv1d_kernel_size[0])
        self.w_2 =  ConvNorm(d_hid, d_in, kernel_size=fft_conv1d_kernel_size[1])

        self.mish = Mish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        residual = input

        output = input.transpose(1, 2)
        output = self.w_2(self.dropout(self.mish(self.w_1(output))))
        output = output.transpose(1, 2)

        output = self.dropout(output) + residual
        return output
    
class SCPositionwiseFeedForward(nn.Module):
    ''' SCCNN and two projection layers '''
    def __init__(self, d_in, d_hid1, d_hid2, dropout=0.1):
        super().__init__()
        self.w_1 = ConvNorm(d_in, d_hid1, kernel_size=1)
        self.w_2 =  ConvNorm(d_hid2, d_in, kernel_size=1)

        self.mish = Mish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, d_w, d_g, d_b, p_w, p_g, p_b):
        '''
        d_w: (B, in_ch, 1, ker)
        d_g: (B, in_ch)
        d_b: (B, in_ch)
        p_w: (B, out_ch, in_ch, 1)
        p_g: (B, out_ch)
        p_b: (B, out_ch)
        default: in_ch = out_ch = 8, ker = 9
        '''
        residual = input
        output = input.transpose(1, 2)
        output = self.dropout(self.mish(self.w_1(output)))
        
        # SC-CNN
        batch = output.size(0)
        p = (d_w.size(-1)-1)//2 #padding
        in_ch = d_w.size(1)
        # weight normalization
        d_w = nn.functional.normalize(d_w, dim=1)*d_g.unsqueeze(-1).unsqueeze(-1)
        p_w = nn.functional.normalize(p_w, dim=1)*p_g.unsqueeze(-1).unsqueeze(-1)
        # convolution
        out = []
        for i in range(batch):
            # Depth-wise
            val = nn.functional.conv1d(output[i].unsqueeze(0),
                                            d_w[i],d_b[i],padding=p,groups=in_ch)
            # Point-wise
            val = nn.functional.conv1d(val,p_w[i],p_b[i])
            out.append(val)
        output = torch.stack(out).squeeze(1)
        
        output = self.dropout(self.mish(output))
        output = self.w_2(output)
        output = output.transpose(1, 2)
        output = self.dropout(output) + residual
        
        return output


class MelStyleEncoder(nn.Module):
    ''' MelStyleEncoder '''
    def __init__(self, config):
        super(MelStyleEncoder, self).__init__()
        self.in_dim = config.n_mel_channels 
        self.hidden_dim = config.style_hidden
        self.out_dim = config.style_vector_dim
        self.kernel_size = config.style_kernel_size
        self.n_head = config.style_head
        self.dropout = config.dropout

        self.spectral = nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout)
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = MultiHeadAttention(self.n_head, self.hidden_dim, 
                                self.hidden_dim//self.n_head, self.hidden_dim//self.n_head, self.dropout) 

        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def forward(self, x, mask=None):
        max_len = x.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None
        
        # spectral
        x = self.spectral(x)
        # temporal
        x = x.transpose(1,2)
        x = self.temporal(x)
        x = x.transpose(1,2)
        # self-attention
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        x, _ = self.slf_attn(x, mask=slf_attn_mask)
        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)

        return w


class Prenet(nn.Module):
    ''' Prenet '''
    def __init__(self, hidden_dim, out_dim, dropout):
        super(Prenet, self).__init__()

        self.convs = nn.Sequential(
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3),
            Mish(),
            nn.Dropout(dropout),
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3),
            Mish(),
            nn.Dropout(dropout),
        )
        self.fc = LinearNorm(hidden_dim, out_dim)

    def forward(self, input, mask=None):
        residual = input
        # convs
        output = input.transpose(1,2)
        output = self.convs(output)
        output = output.transpose(1,2)
        # fc & residual
        output = self.fc(output) + residual

        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)
        return output


class KernelPredictor(nn.Module):
    def __init__(self,
                 style_dim = 128,
                 enc_layer = 4,
                 kernel_size = 9,
                 in_ch = 8,
                 out_ch = 8,
                 ):
        super().__init__()
        
        self.num_layers = enc_layer
        self.ker_size = kernel_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        self.d_w_ch = self.num_layers*kernel_size*in_ch
        self.d_g_ch = self.num_layers*in_ch
        self.d_b_ch = self.num_layers*in_ch
        self.p_w_ch = self.num_layers*in_ch*out_ch
        self.p_g_ch = self.num_layers*out_ch
        self.p_b_ch = self.num_layers*out_ch
    
        self.proj = nn.Linear(style_dim, self.d_w_ch + self.d_g_ch + self.d_b_ch + self.p_w_ch + self.p_g_ch + self.p_b_ch)
    
    def forward(self, x):
        '''
        Extract (direction, gain, bias) for two different types of convolutions.
        '''
        batch = x.size(0)
        x = self.proj(x)
        d_w, d_g, d_b, p_w, p_g, p_b = torch.split(x,[self.d_w_ch, self.d_g_ch, self.d_b_ch, self.p_w_ch, self.p_g_ch, self.p_b_ch], dim=1)
        
        d_w = d_w.contiguous().view(batch, self.num_layers, self.in_ch, 1, self.ker_size)
        d_g = d_g.contiguous().view(batch, self.num_layers, self.in_ch)
        d_b = d_b.contiguous().view(batch, self.num_layers, self.in_ch)
        p_w = p_w.contiguous().view(batch, self.num_layers, self.out_ch, self.in_ch, 1)
        p_g = p_g.contiguous().view(batch, self.num_layers, self.out_ch)
        p_b = p_b.contiguous().view(batch, self.num_layers, self.out_ch)

        return (d_w, d_g, d_b, p_w, p_g, p_b)