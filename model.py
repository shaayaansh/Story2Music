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
        tgt = tgt.permute(1, 0, 2) # (seq_len, batch_size, h_dim)
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=None) # (seq_len, batch_size, h_dim)
        
        output = self.output_layer(out.permute(1, 0, 2)) # (batch_size, seq_len, h_dim)
        
        return output
    
        
    def decoder_forward_only(self, tgt, tgt_mask):
        """
        this function is used for pre-training the decoder
        we pass dummy tensors as the output of the encoder
        """
        tgt_emb = self.midi_embedding(tgt)
        tgt_emb = self.positional_encoder(tgt_emb)
        tgt_emb = tgt_emb.permute(1, 0, 2)
        memory_dummy = torch.zeros((1, tgt_emb.size(1), 768), device=tgt.device)
        out = self.decoder(tgt_emb, memory_dummy, tgt_mask=tgt_mask)
        output = self.output_layer(out.permute(1, 0, 2))
        
        return output

    def generate(self, input_ids,
                 attention_mask,
                 start_token_id,
                 eos_token_id,
                 max_len,
                 decoding_strategy="none",
                 beam_width=4
                ):
        
        device = input_ids.device
        with torch.no_grad():
            memory = self.encoder(input_ids, attention_mask).last_hidden_state
            memory = memory.permute(1, 0, 2)

            start_token = start_token_id
            
            if decoding_strategy == "beam_search":
                generated = self.generate_beam_search(memory, start_token_id, eos_token_id, max_len, beam_width)

            elif decoding_strategy == "top_p":
                generated = self.generate_top_p(
                    memory,
                    start_token_id,
                    eos_token_id,
                    max_len,
                    top_p=0.8
                )
                
            else:
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
    
    
    def generate_beam_search(self, memory, start_token_id, eos_token_id, max_len=100, beam_width=3):
        device = memory.device
        with torch.no_grad():
            beams = [(torch.tensor([start_token_id], device=device).unsqueeze(0), 0)]  # (tokens, score)
            for _ in range(max_len - 1):
                candidates = []
                
                for seq, score in beams:
                    # if there is a stop token, don't expand the beam
                    if seq[0, -1].item() == eos_token_id:
                        candidates.append((seq, score))
                        continue

                    tgt_embedding = self.midi_embedding(seq)
                    tgt_embedding = self.positional_encoder(tgt_embedding)
                    tgt_embedding = tgt_embedding.permute(1, 0, 2)

                    decoder_output = self.decoder(tgt_embedding, memory)
                    output_logits = self.output_layer(decoder_output[-1])  # (batch, vocab)
                    log_probs = torch.log_softmax(output_logits, dim=-1)  # convert to log-probs
                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)

                    for k in range(beam_width):
                        next_token = topk_indices[0, k].unsqueeze(0).unsqueeze(0)  # shape (1, 1)
                        new_seq = torch.cat([seq, next_token], dim=1)
                        new_score = score + topk_log_probs[0, k].item()
                        candidates.append((new_seq, new_score))
                    
                beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

                # Early stopping if all beams end with eos
                if all(seq[0, -1].item() == eos_token_id for seq, _ in beams):
                    break

            best_seq = beams[0][0]
            return best_seq


    def generate_top_p(
        self,
        memory,
        start_token_id,
        eos_token_id,
        max_len=100,
        top_p = 0.8
    ):
        device = memory.device
        with torch.no_grad():
            generated = torch.tensor([start_token_id], dtype=torch.long, device=device)

            for idx in range(max_len-1):
                tgt_embedding = self.midi_embedding(generated)
                tgt_embedding = tgt_embedding.unsqueeze(0) 
                tgt_embedding = self.positional_encoder(tgt_embedding)
                tgt_embedding = tgt_embedding.permute(1, 0, 2)

                decoder_output = self.decoder(memory, tgt_embedding)
                output_logits = self.output_layer(decoder_output[-1])
                next_token = self.top_p_sample(output_logits)

                if next_token.item() == eos_token_id:
                    break

                generated = torch.cat((generated, next_token), dim=0)
        
        return generated


    def top_p_sample(self, logits, p=0.9):
        logits = logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sample_indices = torch.where(cumulative_probs <= p)[0]

        top_probs = sorted_probs[sample_indices]
        top_indices = sorted_indices[sample_indices]

        # re-normalize top-p probability
        top_probs = top_probs / torch.sum(top_probs)

        sampled_index = torch.multinomial(top_probs, num_samples=1)

        return top_indices[sampled_index]

