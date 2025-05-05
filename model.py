import torch
import torch.nn as nn
from transformers import AutoModel
import math
import torch.nn.functional as F


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
    
            

class Story2MusicTransformer(nn.Module):
    def __init__(self, encoder_name, decoder):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name) # download a pretrained text encoder like BERT
        self.positional_encoder = PositionalEncoding(768)
        self.decoder = decoder        
        
    def forward(self, input_ids, attention_mask, tgt, tgt_key_padding_mask=None):
        # freeze the encoder
        with torch.no_grad():
            memory = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        memory = memory.permute(1, 0, 2)  # (T_enc, B, H)

        embeddings = [emb(tgt[:, :, i]) for i, emb in enumerate(self.decoder.embeddings)]
        embeddings_cat = torch.cat(embeddings, dim=-1)  # (B, T, 7*64)
        projected = self.decoder.linear_proj(embeddings_cat)  # (B, T, d_model)
        pos_encoded = self.decoder.pos_encoder(projected).permute(1, 0, 2)  # (T, B, d_model)

        seq_len = pos_encoded.size(0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)

        out = self.decoder.decoder(
            pos_encoded,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,
        )

        out = out.permute(1, 0, 2)  # (B, T, d_model)
        logits = self.decoder.output_proj(out)  # use decoder's own output layer

        return logits
    