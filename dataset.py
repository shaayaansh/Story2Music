from torch.utils.data import Dataset
from transformers import AutoTokenizer
from miditok.pytorch_data import DatasetMIDI, DataCollator
from pathlib import Path
from miditok import REMI, TokenizerConfig
import torch


    
class StoryMidiDataset(Dataset):
      def __init__(self, dataframe, midi_tokenizer):
        self.df = dataframe
        self.midi_tokenizer = midi_tokenizer
        self.midi_paths = "EMOPIA_1.0/midis"
        self.midi_max_length = 1024
        self.pad_token = self.midi_tokenizer["PAD_None"]

      def __len__(self):
        return len(self.df)

      def __getitem__(self, idx):
        input_ids = torch.tensor(self.df.iloc[idx]['tokenized_input_ids'], dtype=torch.long)
        attn_mask = torch.tensor(self.df.iloc[idx]['tokenized_attention_mask'], dtype=torch.long)
        
        midi_id = self.df.iloc[idx]['ID']
        midi_file_path = Path(self.midi_paths, f"{midi_id}.mid")
        midi_tokenized = self.midi_tokenizer(midi_file_path)
        midi_ids = midi_tokenized[0].ids
        
        # Pad/truncate to max_length
        if len(midi_ids) < self.midi_max_length:
            midi_ids += [self.pad_token] * (self.midi_max_length - len(midi_ids))
        else:
            midi_ids = midi_ids[:self.midi_max_length]
            
        midi_token_ids = torch.tensor(midi_ids, dtype=torch.long)

        return input_ids, attn_mask, midi_token_ids