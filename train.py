import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import argparse
from miditok import REMI, TokenizerConfig
from tqdm import tqdm
from pathlib import Path
from model import Story2MusicTransformer
from dataset import StoryMidiDataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset

def ensure_saved_models_dir():
    """
    Check if saved_models directory exists, create it if it doesn't.
    """
    saved_models_dir = Path("saved_models")
    if not saved_models_dir.exists():
        saved_models_dir.mkdir(parents=True)
        print("Created saved_models directory")

def main(args):
    # Ensure saved_models directory exists
    ensure_saved_models_dir()
    
    # hyper parameters TODO: READ FROM ARGS
    vocab_size_midi = 30000  
    batch_size = args.batch_size
    model_name = args.model_name
    num_epochs = args.num_epochs
    lr = args.lr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    # read the training data
    matched_df = pd.read_csv("data/story_midi_matched.csv")


    # initialize the text tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenized_text = tokenizer(
        matched_df['story'].tolist(),
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    matched_df["tokenized_input_ids"] = tokenized_text["input_ids"].tolist()
    matched_df["tokenized_attention_mask"] = tokenized_text["attention_mask"].tolist()


    # initialize the MIDI tokenizer
    tokenizer_params = {
          "pitch_range": (21, 108),  # MIDI range for piano keys
          "beat_res": {(0, 4): 8, (4, 12): 4},
          "num_velocities": 32,
          "special_tokens": ["PAD", "BOS", "EOS", "MASK"]
    }

    config = TokenizerConfig(**tokenizer_params)
    midi_tokenizer = REMI(config)

    pad_token_id = midi_tokenizer["PAD_None"]
    
    # convert dataframe to dataset object
    dataset = StoryMidiDataset(matched_df, midi_tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
    
    
    # instantiate the model & optimizer
    model = Story2MusicTransformer(model_name, vocab_size_midi)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    model.to(device)
    
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for idx, batch in enumerate(tqdm(dataloader)):
            input_ids, attention_mask, midi_output = batch
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            midi_output = midi_output.to(device)
            
            text_input = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            
            # Remove last token from target for input to decoder
            tgt_input = midi_output[:, :-1].to(device)

            # Target for loss should exclude first token (teacher forcing)
            tgt_target = midi_output[:, 1:].to(device)
            
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            optimizer.zero_grad()
            output = model(input_ids, attention_mask, tgt_input, tgt_mask)
            loss = criterion(output.reshape(-1, vocab_size_midi), tgt_target.reshape(-1))
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()

    
    
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        
        
    # === SAVE MODEL ===
    torch.save(model.state_dict(), "saved_models/custom_transformer.pth")
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments to train the story2music model")
    parser.add_argument("--model_name", type=str, required=True, help="name of the pretrained model")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for training the model")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for training the model")
    parser.add_argument("--midi_vocab_size", type=int, default=30000, help="midi vocab size for midi tokenizer")
    parser.add_argument("--num_epochs", type=int, default=20, help="number of epochs to train the model")
    
    args = parser.parse_args()
    main(args)
            
            