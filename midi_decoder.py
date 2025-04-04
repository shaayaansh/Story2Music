import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].detach()
        return x
    


class MidiDecoderOnlyModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, tgt_mask=None, tgt_key_padding_mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x).permute(1, 0, 2)
        out = self.decoder(x, memory=torch.zeros_like(x),
                           tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask
                          )
        
        return self.output(out.permute(1, 0, 2))