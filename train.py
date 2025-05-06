import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import argparse
from miditok import CPWord, TokenizerConfig
from tqdm import tqdm
from pathlib import Path
from model import Story2MusicTransformer
from dataset import StoryMidiDataset
from transformers import BertTokenizer
from cp_transformer import CPWordTransformer
from torch.utils.data import DataLoader, Dataset
import logging
import json
import pickle


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
    
    batch_size = args.batch_size
    model_name = args.model_name
    num_epochs = args.num_epochs
    lr = args.lr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logging.basicConfig(
        filename='training_log.log',        
        level=logging.INFO,                 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

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
    midi_tokenizer = CPWord()
    with open("compound2id.pkl", "rb") as f:
        compound2id = pickle.load(f)
    cp_tokens_size = len(compound2id.items())
    pretrained_decoder = CPWordTransformer(cp_tokens_size, midi_tokenizer)

    checkpoint_path = "pretrain_checkpoints/decoder_epoch_20.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    pretrained_decoder.load_state_dict(checkpoint["model_state_dict"])
    print("Pretrained decoder loaded! \n")

    # convert dataframe to dataset object
    dataset = StoryMidiDataset(matched_df, midi_tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
    print("Dataset item 12: \n")
    print(dataset[12])
    # instantiate the model & optimizer
    model = Story2MusicTransformer(model_name, pretrained_decoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=midi_tokenizer["PAD_None"])
    
    model.to(device)
    print("Story2Music model initiated and moved to device!\n")
    
    logger.info("Training hyperparamteres: \n")
    logger.info(f"Num_epochs: {num_epochs}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Model name: {model_name}")
    logger.info("Training started")
    

    pad_tuple = tuple(midi_tokenizer.pad_token_id for _ in range(len(midi_tokenizer.vocab)))
    compound_pad_id = compound2id[pad_tuple]
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for idx, batch in enumerate(tqdm(dataloader)):
            input_ids, attention_mask, midi_output = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            midi_output = midi_output.to(device) 
            
            decoder_input = midi_output[:, :-1, :]   # (B, T-1, F)
            tgt_target = midi_output[:, 1:, :]  

            B, Tm1, F = tgt_target.shape
            flat = tgt_target.reshape(-1, F).tolist()
            flat_ids = [compound2id[tuple(tok)] for tok in flat]
            tgt_ids = torch.tensor(flat_ids, device=device).view(B, Tm1)

            tgt_key_padding_mask = (tgt_ids == compound_pad_id) 

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(Tm1).to(device)

            optimizer.zero_grad()
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tgt=decoder_input,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            loss = criterion(output.view(-1, output.size(-1)), tgt_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
    
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        
    logger.info("Training finished")   
    # === SAVE MODEL ===
    torch.save(model.state_dict(), "saved_models/custom_transformer.pth")
    logger.info("Model saved")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments to train the story2music model")
    parser.add_argument("--model_name", type=str, required=True, help="name of the pretrained model")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for training the model")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for training the model")
    parser.add_argument("--num_epochs", type=int, default=20, help="number of epochs to train the model")
    
    args = parser.parse_args()
    main(args)
            
            