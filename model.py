import torch
import torch.nn as nn
from transformers import AutoModel
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
    
            

class Story2MusicTransformer(nn.Module):
    def __init__(self, encoder_name, midi_vocab_size):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name) # download a pretrained text encoder like BERT
        self.midi_embedding = nn.Embedding(midi_vocab_size, 768)
        self.positional_encoder = PositionalEncoding(768)
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.output_layer = nn.Linear(768, midi_vocab_size)
        
        
    def forward(self, input_ids, attention_mask, tgt, tgt_mask):
        # freeze text encoder
        with torch.no_grad():
    
            memory = self.encoder(input_ids, attention_mask).last_hidden_state
            memory = memory.permute(1, 0, 2)

            
        tgt = self.midi_embedding(tgt)
        tgt = self.positional_encoder(tgt) # add positional encodings
        tgt = tgt.permute(1, 0, 2) 
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=None)
        
        output = self.output_layer(out)
        return output


    def generate(self, input_ids, attention_mask, max_len, start_token_id, eos_token_id):
        device = input_ids.device
        with torch.no_grad():
            memory = self.encoder(input_ids, attention_mask).last_hidden_state
            memory = memory.permute(1, 0, 2)

            start_token = start_token_id
            generated = torch.tensor([[start_token]], dtype=torch.long, device=device)

            for idx in range(max_len-1):
                tgt_embedding = self.midi_embedding(generated)
                tgt_embedding = self.positional_encoder(tgt_embedding)
                tgt_embedding = tgt_embedding.permute(1, 0, 2)

                decoder_output = self.decoder(tgt_embedding, memory, tgt_mask=None, memory_mask=None)
                output_logits = self.output_layer(decoder_output[-1])
                next_token = torch.argmax(output_logits, dim=-1).unsqueeze(0)

                if next_token.item() == eos_token_id:
                    break

                generated = torch.cat((generated, next_token), dim=1)
            

        return generated
                

