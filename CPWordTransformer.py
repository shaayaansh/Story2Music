import torch.nn.functional as F
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


class CPWordTransformer(nn.Module):
  def __init__(self, cp_tokens_size, tokenizer, num_layers= 6, d_model=128):
    super().__init__()
    self.tokenizer = tokenizer
    self.embeddings = nn.ModuleList([
        nn.Embedding(len(vocab), 64) for vocab in self.tokenizer.vocab
    ])


    decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
    self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
    self.linear_proj = nn.Linear(len(tokenizer.vocab)*64, d_model)
    self.pos_encoder = PositionalEncoding(d_model)
    self.output_proj = nn.Linear(d_model, cp_tokens_size)

  def forward(self, x, tgt_key_padding_mask=None):
    """
    input to the function is (B, T, 7)
    """
    embeddings = [embd(x[:,:,i]) for i, embd in enumerate(self.embeddings)]
    embeddings_cat = torch.cat(embeddings, dim=-1) # super token (B, T, 7*64)
    projected = self.linear_proj(embeddings_cat) # (B, T, d_model)
    pos_encoded = self.pos_encoder(projected).permute(1, 0, 2) # (T, B, d_model)

    seq_len = pos_encoded.size(0)
    tgt_mask = torch.triu(torch.ones(seq_len, seq_len,
                                     device=x.device,
                                     dtype=torch.bool))

    out = self.decoder(pos_encoded,
                       memory=torch.zeros_like(pos_encoded),
                       tgt_mask=tgt_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask
                       )

    out = out.permute(1, 0, 2) # (B, T, d_model)
    out = self.output_proj(out)

    return out


  def generate(self,
                 start_feats,
                 eos_feats,
                 eos_token_id,
                 max_len,
                 compound2id,
                 id2compound,
                 decoding_strategy="top_p",
                 top_p=0.9,
                 device=None
                ):

        device = device
        start_token = compound2id[start_feats]
        eos_id = compound2id[eos_feats]

        generated_feats = [start_feats]

        with torch.no_grad():
              for _ in range(max_len - 1):
                x = torch.tensor(
                    [generated_feats],
                    device=device,
                    dtype=torch.long
                ) # (1, T, F)

                logits = self.forward(x)
                last_logits = logits[0, -1]

                if decoding_strategy == "greedy":
                    next_id = int(last_logits.argmax())
                    
                elif decoding_strategy == "top_p":
                    next_id = int(self.top_p_sample(logits, p=top_p))
                
                generated_feats.append(id2compound[next_id])

                if next_id == eos_id:
                  break

        return [compound2id[t] for t in generated_feats]

  
  def top_p_sample(self, logits, p=0.9):
      probs = F.softmax(logits, dim=-1)
      sorted_probs, sorted_idx = torch.sort(probs, descending=True)
      cum_probs = torch.cumsum(sorted_probs, dim=-1)

      mask = cum_probs <= p
      mask[..., 0] = True # include at least the first token

      filtered_probs = sorted_probs[mask]
      filtered_idx = sorted_idx[mask]

      filtered_probs = filtered_probs / filtered_probs.sum() # normalize the filtered probabilities

      choice = torch.multinomial(filtered_probs, 1)
      return filtered_idx[choice]