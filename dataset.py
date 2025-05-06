from torch.utils.data import Dataset
from transformers import AutoTokenizer
from miditok.pytorch_data import DatasetMIDI, DataCollator
from pathlib import Path
from miditok import CPWord, TokenizerConfig
import torch


class StoryMidiDataset(Dataset):
    def __init__(self, dataframe, midi_tokenizer, max_length=512):
        self.df = dataframe
        self.midi_tokenizer = midi_tokenizer
        self.midi_paths = "EMOPIA_1.0/midis"
        self.midi_max_length = max_length
        self.pad_token = tuple(midi_tokenizer.pad_token_id for _ in range(len(midi_tokenizer.vocab)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.df.iloc[idx]['tokenized_input_ids'], dtype=torch.long)
        attn_mask = torch.tensor(self.df.iloc[idx]['tokenized_attention_mask'], dtype=torch.long)
        
        midi_id = self.df.iloc[idx]['ID']
        midi_file_path = Path(self.midi_paths, f"{midi_id}.mid")
        midi_tokenized = self.midi_tokenizer(midi_file_path)
        midi_ids = midi_tokenized[0].ids  # List[List[int]] (T, F)

        # Pad or truncate
        midi_tensor = torch.tensor(midi_ids, dtype=torch.long)
        T, F = midi_tensor.shape
        if T < self.midi_max_length:
            pad_token = [self.midi_tokenizer.pad_token_id] * midi_tensor.shape[1]
            pad_tensor = torch.tensor([pad_token] * (self.midi_max_length - T), dtype=torch.long)
            midi_tensor = torch.cat([midi_tensor, pad_tensor], dim=0)
        else:
            midi_tensor = midi_tensor[:self.midi_max_length]

        return input_ids, attn_mask, midi_tensor  # shapes: (L,), (L,), (T, F)