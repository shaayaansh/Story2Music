from model import Story2MusicTransformer
import torch
import argparse
from miditok import REMI, TokenizerConfig
from tqdm import tqdm
from pathlib import Path
from model import Story2MusicTransformer
from dataset import StoryMidiDataset
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# Load model
model = Story2MusicTransformer("bert-base-uncased", midi_vocab_size=30000)
#model.load_state_dict(torch.load("saved_models/custom_transformer.pth"))
model.load_state_dict(torch.load("saved_models/bert-base-60-epochs.pth"))
model.eval()

tokenizer_params = {
    "pitch_range": (21, 108),  # MIDI range for piano keys
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"]
}

config = TokenizerConfig(**tokenizer_params)
midi_tokenizer = REMI(config)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

story = "It was a nice day and everyone was smiling at the park it was just amazing."

inputs = tokenizer(story, return_tensors="pt", truncation=True, padding=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]


# Generate
print("GENERATING USING BEAM SEARCH")
generated = model.generate(input_ids, attention_mask, 4, 3, max_len=50, decoding_strategy="beam_search", beam_width=4)
print("Generated MIDI:", generated.squeeze().tolist())


print("GENERATING WITHOUT USING BEAM SEARCH")
generated = model.generate(input_ids, attention_mask, 4, 3, max_len=50)
print("Generated MIDI:", generated.squeeze().tolist())