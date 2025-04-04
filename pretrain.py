import torch.optim as optim
import torch
import torch.nn as nn
from pathlib import Path
from miditok import REMI
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
from utils import generate_causal_mask, load_and_split_pretraining_data
from random import shuffle
from miditok.data_augmentation import augment_dataset
from midi_decoder import MidiDecoderOnlyModel
from tqdm import tqdm
import os


def main():
    
    os.makedirs("pretrain_checkpoints", exist_ok=True)
    
    midi_tokenizer = REMI()
    
    # load and split training data (only run it once for the first time)
    #load_and_split_pretraining_data(midi_tokenizer)
        
    midi_paths = list(Path("dataset_train").resolve().glob("**/*.mid"))

    dataset = DatasetMIDI(
        files_paths=midi_paths,
        tokenizer=midi_tokenizer,
        max_seq_len=1024,
        bos_token_id=midi_tokenizer.pad_token_id,
        eos_token_id=midi_tokenizer["BOS_None"],
    )
    
    collator = DataCollator(midi_tokenizer.pad_token_id)
    data_loader = DataLoader(dataset=dataset, collate_fn=collator, batch_size=8)

    model = MidiDecoderOnlyModel(vocab_size=len(midi_tokenizer.vocab))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=midi_tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    
    num_epochs = 601
    save_every = 200

    model.train()
    for epoch in range(num_epochs):
        for _, batch in enumerate(tqdm(data_loader)):
            input_ids = batch['input_ids'].to(device)            # (batch_size, seq_len)
            attention_mask = batch['attention_mask'].to(device)  # (batch_size, seq_len)

            decoder_input = input_ids[:, :-1]        # (batch_size, seq_len - 1)
            target = input_ids[:, 1:]                # (batch_size, seq_len - 1)
            attn_mask = attention_mask[:, :-1]       # (batch_size, seq_len - 1)

            tgt_mask = generate_causal_mask(decoder_input.size(1)).to(device)
            tgt_key_padding_mask = (attn_mask == 0)

            output = model(
                decoder_input,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )  # (batch_size, seq_len - 1, vocab)

            loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        print(f"Epoch {epoch+1} — Loss: {loss.item():.4f}")

        if epoch % save_every == 0 and epoch != 0:
            checkpoint_path = f"pretrain_checkpoints/decoder_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    
    

if __name__ == "__main__":
    main()
