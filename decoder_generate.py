import torch
import torch.nn as nn
from cp_transformer import CPWordTransformer
import argparse
from miditok import CPWord
from collections import OrderedDict
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = CPWord()
start_feats = tuple(mapping["BOS_None"] for mapping in tokenizer.vocab)
eos_feats   = tuple(mapping["EOS_None"] for mapping in tokenizer.vocab)

with open("compound2id.pkl", "rb") as f:
    compound2id = pickle.load(f)

with open("id2compound.pkl", "rb") as f:
    id2compound = pickle.load(f)


model = CPWordTransformer(len(compound2id.items()), tokenizer)
model.to(device)

checkpoint_path = "pretrain_checkpoints/decoder_epoch_4.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

new_state_dict = OrderedDict()
for key, value in checkpoint['model_state_dict'].items():
    new_key = key.replace("module.", "")
    new_state_dict[new_key] = value



model.load_state_dict(new_state_dict)
print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
model.eval()


print("GENERATING USING GREEDY SAMPLING")

generated_ids = model.generate(
    start_feats=start_feats,
    eos_feats=eos_feats,
    eos_token_id=None,
    max_len=50,
    compound2id=compound2id,
    id2compound=id2compound,
    decoding_strategy="greedy",
    device=device
)
print("Generated MIDI:", generated_ids)


print("==="*30)
print("GENERATING USING TOP P SAMPLING")
generated_ids = model.generate(
    start_feats=start_feats,
    eos_feats=eos_feats,
    eos_token_id=None,
    max_len=1024,
    compound2id=compound2id,
    id2compound=id2compound,
    decoding_strategy="top_p",
    device=device
)
print("Generated MIDI:", generated_ids)



