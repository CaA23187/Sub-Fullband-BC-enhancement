import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import numpy as np
from functools import partial

import torch.utils
import torch.utils.flop_counter

from sConformer_Causal import sConformer


class Conv2dNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, fpad: bool = True, separable=False, dilation=1, time_dilation=1, bias=True, act=nn.Hardswish) -> None:
        super().__init__()

        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        if max(kernel) == 1:
            separable = False

        models = []
        if fpad:
            fpad_ = kernel[1] // 2 + dilation - 1
        else:
            fpad_ = 0
        pad = (fpad_, fpad_, kernel[0] + time_dilation-1 - 1, 0)
        models.append(nn.ConstantPad2d(pad, 0.))
        models.append(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                stride=(1, stride),  # Stride over time is always 1
                dilation=(time_dilation, dilation),  # Same for dilation
                groups=groups,
                bias=bias,
            )
        )
        if separable:
            models.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
        models.append(nn.BatchNorm2d(out_ch))
        if act is not None:
            models.append(act())

        self.conv = nn.Sequential(*models)

    def forward(self, x):
        x = self.conv(x)
        return x
class ConvTranspose2dNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, fpad=True, output_padding=0, separable=False, dilation=1, time_dilation=1, bias=True, act=nn.Hardswish) -> None:
        super().__init__()

        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        if max(kernel) == 1:
            separable = False

        models = []
        if fpad:
            fpad_ = kernel[1] // 2
        else:
            fpad_ = 0
        pad = (-fpad_, -fpad_, 0, -(kernel[0] + time_dilation-1 - 1))   # (F_low, F_high, T_low, T_high) !!!!!
        models.append(
            nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                # padding=(kernel[0] - 1, fpad_ + dilation - 1),
                output_padding=(0, output_padding),
                stride=(1, stride),  # Stride over time is always 1
                dilation=(time_dilation, dilation),
                groups=groups,
                bias=bias,
            )
        )
        models.append(nn.ConstantPad2d(pad, 0.))

        if separable:
            models.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
        models.append(nn.BatchNorm2d(out_ch))
        if act is not None:
            models.append(act())

        self.conv = nn.Sequential(*models)

    def forward(self, x):
        x = self.conv(x)
        return x
       

class GroupedsConformer(nn.Module):
    def __init__(
        self,input_dim,num_heads,ffn_dim,num_layers=2,groups=2, rearrange=False, rearrange_first=False, depthwise_conv_kernel_size=31,dropout=0.1,causal=False,lookahead=0, 
    ):
        super().__init__()

        assert input_dim % groups == 0, (input_dim, groups)
        input_dim_t = input_dim // groups

        self.groups = groups
        self.num_layers = num_layers
        self.rearrange = rearrange
        self.first_rearrange = rearrange_first

        self.lstm_list = nn.ModuleList()
        for _ in range(num_layers):
            self.lstm_list.append(
                nn.ModuleList(
                    [
                        sConformer(input_dim_t,num_heads,ffn_dim,num_layers=1,
                                   depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                                   dropout=dropout,
                                   causal=causal,
                                   lookahead=lookahead)
                        for _ in range(groups)
                    ]
                )
            )

    def forward(self, x):
        """Grouped LSTM forward.

        Args:
            x (torch.Tensor): (B, C, T, D)
        Returns:
            out (torch.Tensor): (B, C, T, D)
        """
        B, C, T, D = x.shape
        out = x
        out = out.transpose(1, 2).contiguous()
        # out = out.transpose(1, 2).to(memory_format=torch.contiguous_format)
        out = out.reshape(B, T, -1)


        for idx, lstms in enumerate(self.lstm_list):
            if self.first_rearrange:
                out = (
                        out.reshape(B, T, self.groups, -1)
                        .transpose(-1, -2) 
                        .contiguous()
                        # .to(memory_format=torch.contiguous_format)
                        .reshape(B, T, -1)
                    )
            _out = torch.chunk(out, self.groups, dim=-1)

            out = torch.cat(
                [lstm(_out[i])[0] for i, lstm in enumerate(lstms)],
                dim=-1,
            )

            # out = self.ln[0](out)
            
            if self.rearrange:
                if idx < (self.num_layers-1):
                    out = (
                        out.reshape(B, T, self.groups, -1)
                        .transpose(-1, -2) 
                        .contiguous()
                        # .to(memory_format=torch.contiguous_format)
                        .reshape(B, T, -1)
                    )
            
        out = out.view(B, T, C, -1)
        out = out.transpose(1, 2).contiguous()
        # out = out.transpose(1, 2).to(memory_format=torch.contiguous_format)

        return out

class iAG(nn.Module):
    '''
    improved Attention Gate
    '''
    def __init__(self, ch, ch1, ch2):
        super(iAG, self).__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(ch, 1, 1),
            # nn.BatchNorm2d(1),
        )
        self.global_conv = nn.Sequential(
            nn.Conv2d(ch, 1, 1),
            # nn.BatchNorm2d(1),
        )
        self.pwconv = nn.Conv2d(ch, ch, 1)
    
    def forward(self, x1, x2):
        '''
        x1: decoder, gate
        x2: encoder
        '''
        x2_in = x2.clone()

        x = x1+x2
        x_local = self.local_conv(x)
        x_global = self.global_conv(torch.mean(x, dim=-1, keepdim=True))
        x = torch.sigmoid(x_local+x_global) * x2_in
        x = self.pwconv(x)
        return x

class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''
    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(inter_channels),   
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),    
            nn.BatchNorm2d(channels),
        )
        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(inter_channels),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(inter_channels),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(inter_channels),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, residual):
        # print('x in iAFF',x.shape)
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

class DeepFitler(nn.Module):
    """Deep Filtering."""
    def __init__(self, num_freqs: int, frame_size: int = 5,lookahead: int = 0, conj: bool = False):
        super().__init__()
        self.num_freqs = num_freqs
        self.frame_size = frame_size
        self.lookahead = lookahead
        self.conj = conj

        assert frame_size >= lookahead + 1, 'lookahead > frame_size - 1'
        padding = (0,0,frame_size-1 - lookahead,lookahead)  # (F_low, F_high, T_low, T_high)
        self.pad = nn.ConstantPad2d(padding, 0.)

    def forward(self, spec: Tensor, coefs: Tensor, alpha= None) -> Tensor:
        # spec (real) [B, 2, T, F], O: df_order
        # coefs (real) [B, 2*O, T, F]
        assert coefs.shape[1] == self.frame_size*2, "order not match!"

        B, C, T, F = coefs.shape
        padded = self.pad(spec[..., : self.num_freqs])
        padded = padded.unfold(dimension=2, size=self.frame_size, step=1) # [B, 2, T, F, O]

        # coefs = coefs.view([B, 2, self.frame_size, T, F]).permute(0,1,3,4,2).contiguous() # [B, 2, T, F, O] .to(memory_format=torch.contiguous_format)
        coefs = coefs.view([B, 2, self.frame_size, T, F]).permute(0,1,3,4,2).contiguous()

        spec_f_real = padded[:,0,...] * coefs[:,0,...] - padded[:,1,...] * coefs[:,1,...]
        spec_f_imag = padded[:,1,...] * coefs[:,0,...] + padded[:,0,...] * coefs[:,1,...]
        spec_f = torch.stack([spec_f_real, spec_f_imag], dim=1)
        spec_f = spec_f.sum(dim=-1)

        spec = torch.cat([spec_f, spec[..., :, self.num_freqs:]], dim=-1)
        return spec


class Sub_Full_band(nn.Module):
    def __init__(self, conv_ch=16, conv_kernel_inp=(3,3), conv_kernel=(1,3), conv_lookahead=0, n_fft=320, sr=16000, nb_erb=48, min_nb_freqs=1, df_bins=96, df_order=3, df_lookahead=0, conv_bias=False, iaff=True, fpad=False, causal=True) -> None:
        super(Sub_Full_band, self).__init__()
        self.n_fft = n_fft
        self.hop_len = n_fft//2

        self.df_order = df_order
        self.df_bins = df_bins
        self.df_lookahead = df_lookahead
        
        ERB_width = self.cal_erb_width(sr, nb_erb, n_fft, min_nb_freqs)
        # print("ERB_width: ",ERB_width)
        erb_fb = self.ERB_fb(ERB_width, sr, inverse=False)
        ERB_inverse = self.ERB_fb(ERB_width, sr, inverse=True)
        self.register_buffer("erb_fb", erb_fb)
        self.register_buffer("erb_inv_fb", ERB_inverse)
        # print(ERB_width)
        # print("erb_fb: ", erb_fb)
        # print("erb_inv_fb: ", ERB_inverse)
        
        self.iaff = iaff
        if iaff:
            self.iaff_1 = iAFF(channels=1, r=1)
            self.iaff_2 = iAFF(channels=2, r=1)

        pad = (0, 0, -conv_lookahead, conv_lookahead)
        self.pad = nn.ConstantPad2d(pad, 0.)
        
        ## erb encoder
        self.erb_enc = nn.ModuleList()
        self.erb_enc.append(Conv2dNormAct(3, conv_ch, kernel=conv_kernel_inp, bias=conv_bias, separable=False, stride=1, fpad=False))
        self.erb_enc.append(Conv2dNormAct(conv_ch, conv_ch, kernel=conv_kernel, bias=conv_bias, separable=False, stride=2, fpad=False))
        self.erb_enc.append(Conv2dNormAct(conv_ch, conv_ch, kernel=conv_kernel, bias=conv_bias, separable=False, stride=2, fpad=False))
        # self.erb_conv.append(Conv2dNormAct(conv_ch, conv_ch, kernel=conv_kernel, bias=False, separable=False, stride=2, fpad=False))

        ## erb encoder BC
        self.erb_BC_enc = nn.ModuleList()
        self.erb_BC_enc.append(Conv2dNormAct(1, conv_ch, kernel=conv_kernel_inp, bias=conv_bias, separable=False, stride=1, fpad=False))
        self.erb_BC_enc.append(Conv2dNormAct(conv_ch, conv_ch, kernel=conv_kernel, bias=conv_bias, separable=False, stride=2, fpad=False))
        self.erb_BC_enc.append(Conv2dNormAct(conv_ch, conv_ch, kernel=conv_kernel, bias=conv_bias, separable=False, stride=2, fpad=False))
        # self.erb_BC_conv.append(Conv2dNormAct(conv_ch, conv_ch, kernel=conv_kernel, bias=False, separable=False, stride=2, fpad=False))

        self.iaff_3 = iAFF(channels=conv_ch, r=4)

        ## df encoder
        self.df_enc = nn.ModuleList()
        self.df_enc.append(Conv2dNormAct(4, conv_ch, kernel=conv_kernel_inp, bias=conv_bias, separable=False, stride=2, fpad=False))
        self.df_enc.append(Conv2dNormAct(conv_ch, conv_ch, kernel=conv_kernel, bias=conv_bias, separable=False, stride=2, fpad=False))
        # self.df_enc.append(Conv2dNormAct(conv_ch, conv_ch, kernel=conv_kernel_inp, bias=conv_bias, separable=False, stride=2, fpad=False)) // 曾经搞错了!
        self.df_enc.append(Conv2dNormAct(conv_ch, conv_ch, kernel=conv_kernel, bias=conv_bias, separable=False, stride=2, fpad=False))
        self.df_enc.append(Conv2dNormAct(conv_ch, conv_ch, kernel=conv_kernel, bias=conv_bias, separable=False, stride=2, fpad=False))
        self.df_enc.append(Conv2dNormAct(conv_ch, conv_ch, kernel=conv_kernel, bias=conv_bias, separable=False, stride=2, fpad=False))
        
        bottle_dim = 6 ## 根据Encoder最后一层的维度修改
        # self.groupGRU1 = GroupedGRU(bottle_dim * conv_ch * 2, num_layers=1, groups=2, rearrange=False)
        self.groupGRU1 = GroupedsConformer(bottle_dim * conv_ch * 2, num_heads=4, ffn_dim=int(1* bottle_dim * conv_ch * 2), num_layers=1, groups=4, rearrange=False, rearrange_first=False, causal=causal, dropout=0.1, depthwise_conv_kernel_size=7)

        ## erb decoder
        # self.groupGRU2 = GroupedGRU(bottle_dim * conv_ch * 2, num_layers=1, groups=2, rearrange=True, first_rearrange=True)
        self.groupGRU2 = GroupedsConformer(bottle_dim * conv_ch * 2, num_heads=4, ffn_dim=int(1* bottle_dim * conv_ch * 2), num_layers=1, groups=4, rearrange=True, rearrange_first=True, causal=causal, dropout=0.1, depthwise_conv_kernel_size=7)
        self.pwconv = nn.Conv2d(conv_ch * 2, conv_ch, 1)
        self.erb_dec = nn.ModuleList()
        # self.erb_deconv.append(ConvTranspose2dNormAct(conv_ch*2, conv_ch, kernel=conv_kernel, bias=False, separable=True, stride=2, fpad=fpad, output_padding=1))
        self.erb_dec.append(ConvTranspose2dNormAct(conv_ch*2, conv_ch, kernel=conv_kernel, bias=conv_bias, separable=True, stride=2, fpad=False, output_padding=1))
        self.erb_dec.append(ConvTranspose2dNormAct(conv_ch*2, conv_ch, kernel=conv_kernel, bias=conv_bias, separable=True, stride=2, fpad=False, output_padding=1))
        self.erb_dec.append(ConvTranspose2dNormAct(conv_ch*2, 1, kernel=conv_kernel_inp, bias=conv_bias, separable=True, stride=1, fpad=False, output_padding=0))

        # ## skip
        self.skip_erb = nn.ModuleList()
        pconv = partial(nn.Conv2d, kernel_size=1) ## pwconv
        self.skip_erb.append(iAG(conv_ch, conv_ch, conv_ch))
        self.skip_erb.append(iAG(conv_ch, conv_ch, conv_ch))
        self.skip_erb.append(iAG(conv_ch, conv_ch, conv_ch))
        # self.skip.append(iAG(conv_ch, conv_ch, conv_ch))

        # ## df decoder
        # self.groupGRU3 = GroupedGRU(bottle_dim * conv_ch * 2, num_layers=2, groups=2, rearrange=True, first_rearrange=True)
        # self.groupGRU3 = GroupedsConformer(bottle_dim * conv_ch * 2, num_heads=4, ffn_dim=int(1* bottle_dim * conv_ch * 2), num_layers=2, groups=4, rearrange=True, rearrange_first=True, causal=causal, dropout=0.1, depthwise_conv_kernel_size=7)
        self.groupGRU3 = GroupedsConformer(bottle_dim * conv_ch * 2, num_heads=4, ffn_dim=int(1* bottle_dim * conv_ch * 2), num_layers=1, groups=4, rearrange=True, rearrange_first=True, causal=causal, dropout=0.1, depthwise_conv_kernel_size=7)
        self.pwconv_df = pconv(conv_ch * 2, conv_ch)

        self.df_dec = nn.ModuleList()
        self.df_dec.append(ConvTranspose2dNormAct(conv_ch*2, conv_ch, kernel=conv_kernel, bias=conv_bias, separable=True, stride=2, fpad=False, output_padding=0))
        self.df_dec.append(ConvTranspose2dNormAct(conv_ch*2, conv_ch, kernel=conv_kernel, bias=conv_bias, separable=True, stride=2, fpad=False, output_padding=0))
        self.df_dec.append(ConvTranspose2dNormAct(conv_ch*2, conv_ch, kernel=conv_kernel, bias=conv_bias, separable=True, stride=2, fpad=False, output_padding=0))
        self.df_dec.append(ConvTranspose2dNormAct(conv_ch*2, conv_ch, kernel=conv_kernel, bias=conv_bias, separable=True, stride=2, fpad=False, output_padding=1))
        self.df_dec.append(ConvTranspose2dNormAct(conv_ch*2, df_order*2, kernel=conv_kernel_inp, bias=conv_bias, separable=True, stride=2, fpad=False, output_padding=1))

        # ## skip DF
        self.skip_df = nn.ModuleList()
        pconv = partial(nn.Conv2d, kernel_size=1) ## pwconv
        self.skip_df.append(pconv(conv_ch, conv_ch))
        self.skip_df.append(pconv(conv_ch, conv_ch))
        self.skip_df.append(pconv(conv_ch, conv_ch))
        self.skip_df.append(pconv(conv_ch, conv_ch))
        self.skip_df.append(pconv(conv_ch, conv_ch))

        
        self.df_op = DeepFitler(df_bins, frame_size=df_order, lookahead=df_lookahead)

        # self.high_mask = nn.Linear(161,161)
        

    def preproc(self, audio):
        ## STFT and compress
        spec = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_len, window=torch.hann_window(self.n_fft, device=audio.device), return_complex=True).transpose(-1,-2)
        mag, phase = spec.abs(), spec.angle()
        spec = (mag**0.5) * torch.exp(1j*phase)
        # print('spec ', spec.shape)
        ## transform to ERB
        ERB_matrx = self.erb_fb
        ERB = torch.matmul(mag**0.5, ERB_matrx)
        # ERB = 10*torch.log10(ERB+1e-8)

        erb_feat = ERB.unsqueeze(1)
        spec_feat = spec[..., :self.df_bins].unsqueeze(1)
        spec = spec.unsqueeze(1)

        # spec = torch.view_as_real(spec)
        # spec_feat = torch.view_as_real(spec_feat)
        spec = torch.cat([spec.real, spec.imag], dim=1)
        spec_feat = torch.cat([spec_feat.real, spec_feat.imag], dim=1)
        # print(spec.shape, erb_feat.shape, spec_feat.shape)
        return spec, erb_feat, spec_feat
    
    ## featrue extraction
    @torch.no_grad()
    def ERB_fb(self, widths: np.ndarray, sr: int, normalized: bool = True, inverse: bool = False) -> Tensor:
        '''
        obtain ERB transform matrix
        '''
        n_freqs = int(np.sum(widths))
        all_freqs = torch.linspace(0, sr // 2, n_freqs + 1)[:-1]

        b_pts = np.cumsum([0] + widths.tolist()).astype(int)[:-1]

        fb = torch.zeros((all_freqs.shape[0], b_pts.shape[0]))
        for i, (b, w) in enumerate(zip(b_pts.tolist(), widths.tolist())):
            fb[b : b + w, i] = 1
        # Normalize to constant energy per resulting band
        if inverse:
            fb = fb.t()
            if not normalized:
                fb /= fb.sum(dim=1, keepdim=True)
        else:
            if normalized:
                fb /= fb.sum(dim=0)
        # return fb.to(device=get_device())
        return fb
    @torch.no_grad()
    def cal_erb_width(self, sr, n_erb, n_fft, min_nb_freqs):
        def Hz2erb_other(f):
            """ Convert Hz to ERB number """
            n_erb = 21.4 * np.log10(1 + 0.00437 * f)
            return n_erb
        def erb2Hz_other(e):
            """ Convert ERB to Hz number """
            f = (10 ** (e/21.4) -1) / 0.00437
            return f

        erb_ = Hz2erb_other(sr//2)
        erb = np.linspace(0, erb_, n_erb + 1)
        erb_band = erb2Hz_other(erb[1:])
        erb_band = np.concatenate([np.array([0]), erb_band])

        resolution = sr/n_fft

        erb_width = np.empty(n_erb)

        current_f = 0
        for i in range(n_erb):
            if (erb_band[i+1] - current_f)<min_nb_freqs*resolution:
                erb_width[i] = min_nb_freqs
                current_f = current_f + min_nb_freqs*resolution
            else:
                width = (erb_band[i+1] - current_f) / resolution
                width = int(np.round(width).item())
                erb_width[i] = width
                current_f = current_f + width*resolution
        erb_width[-1] = erb_width[-1] +1    

        erb_width = erb_width.astype(np.int32)
        return erb_width

    def forward(self, x):
        '''
        x: shape [B, 2, samples]
        '''
        with torch.no_grad():
            ac, bc = x.chunk(chunks=2, dim=1)
            ac = ac.squeeze(1)
            bc = bc.squeeze(1)
            spec, feat_erb_ac, feat_spec_ac = self.preproc(ac)
            _, feat_erb_bc, feat_spec_bc = self.preproc(bc)

        '''
        spec: [B, C=2, T, F], 
        feat_erb:  [B, C=1, T, erb=32]
        feat_spec: [B, C=2, T, F=96] (48 kHz)
        '''
        # print(spec.shape, feat_erb_ac.shape, feat_spec_ac.shape)

        if self.iaff:
            fuse = self.iaff_1(feat_erb_ac, feat_erb_bc)
            feat_erb = torch.cat([fuse, feat_erb_ac, feat_erb_bc], dim=1)
            # print("feat_erb: ", feat_erb.shape)

            fuse = self.iaff_2(feat_spec_ac, feat_spec_bc)
            # feat_spec = torch.cat([fuse, feat_spec_ac, feat_spec_bc], dim=1)
            feat_spec = torch.cat([fuse, feat_spec_ac], dim=1)
            # print("feat_spec: ", feat_spec.shape)
            

        ## input padding
        feat_erb = self.pad(feat_erb)
        feat_spec = self.pad(feat_spec)
        
        # print('erb enc_in: ', feat_erb.shape)
        ## encoder ERB Fused AC-BC
        enc_out = []
        for enc in self.erb_enc:
            feat_erb = enc(feat_erb)
            enc_out.append(feat_erb)
            # print('erb enc_out: ', feat_erb.shape)

        # print('erb enc_BC_in: ', feat_erb_bc.shape)
        ## encoder ERB BC
        enc_BC_out = []
        for enc in self.erb_BC_enc:
            feat_erb_bc = enc(feat_erb_bc)
            enc_BC_out.append(feat_erb_bc)
            # print('erb enc_BC_out: ', feat_erb_bc.shape)

        # print('df enc_out: ', feat_spec.shape)
        ## encoder Spec Fused AC-BC
        df_out = []
        for enc in self.df_enc:
            feat_spec = enc(feat_spec)
            df_out.append(feat_spec)
            # print('df enc_out: ', feat_spec.shape)
        
        ## inter fusion AC-BC and BC
        feat_erb = self.iaff_3(feat_erb, feat_erb_bc)
        feat_erb = torch.cat([feat_erb, feat_spec], dim=1)

        ## bottlenecks
        # print("bottlenecks in: ", feat_erb.shape)
        feat_erb = self.groupGRU1(feat_erb)
        bottle = feat_erb.clone()
        # print("bottlenecks out: ", feat_erb.shape)

        # print("bottlenecks in: ", feat_erb.shape)
        feat_erb = self.groupGRU2(feat_erb)
        feat_erb = self.pwconv(feat_erb)
        # print("bottlenecks out: ", feat_erb.shape)

        # print("bottlenecks in: ", bottle.shape)
        df_coef = self.groupGRU3(bottle)
        df_coef = self.pwconv_df(df_coef)
        # print("bottlenecks out: ", df_coef.shape)
        
        ## decoder erb
        for idx, (dec, skip) in enumerate(zip(self.erb_dec, self.skip_erb)):
            res = skip(feat_erb, enc_out[-(idx+1)])
            feat_erb = torch.cat([feat_erb, res], dim=1)
            # print('erb dec in: ', feat_erb.shape)
            feat_erb = dec(feat_erb)
            # print('erb dec out: ', feat_erb.shape)
        feat_erb = torch.sigmoid(feat_erb)
        
        
        ## decoder df
        for idx, (dec, skip) in enumerate(zip(self.df_dec, self.skip_df)):
            res = skip(df_out[-(idx+1)])
            df_coef = torch.cat([df_coef, res], dim=1)
            # print('df dec in: ', df_coef.shape)
            df_coef = dec(df_coef)
            # print('df dec out: ', df_coef.shape)

        
        ## apply ERB mask
        feat_erb = torch.matmul(feat_erb, self.erb_inv_fb)
        # print("feat_erb: ", feat_erb.shape)
        spec = spec * feat_erb
        # print('Spec_erb: ', spec.shape)

        ## deep filter
        spec = self.df_op(spec, df_coef)
        # spec_f = spec[...,:self.df_bins]* df_coef
        # spec = torch.cat([spec_f, spec[...,self.df_bins:]], dim=-1)
        # spec = self.high_mask(spec)
        return spec


def get_Num_parameter(model):
    total = sum([param.nelement() for param in model.parameters()])
    trainable_num = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Total parameters: {total/1e6:.4f}M, Trainable: {trainable_num/1e6:.4f}M" )

if __name__ == '__main__':
    dev = 'cuda:0'
    net = Sub_Full_band(
        n_fft=512,
        conv_ch=12,
        conv_kernel_inp=(2,4),
        conv_kernel = (1,4),
        conv_lookahead=0,
        nb_erb=36,
        min_nb_freqs=1,
        df_bins=257,
        df_order=3,
        df_lookahead=0,
        conv_bias=False,
        iaff=True,
        fpad=True,
        causal=True,
    ).to(dev)

    net.eval()
    get_Num_parameter(net)
    data = torch.rand([2, 2, 64000]).to(dev)
    out = net(data)
    print('out: ',out.shape)

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(net, (2, 64000), as_strings=True,
                                           print_per_layer_stat=False, verbose=False, output_precision=5)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print(f'{net._get_name()}, Real MACs: ', float(macs.split(' ')[0])/4)
    