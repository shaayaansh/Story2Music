import torch.optim as optim
import torch
import torch.nn as nn
from pathlib import Path
from miditok import CPWord, TokenizerConfig
from symusic import Score
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
from utils import generate_causal_mask
from utils import load_pretrain_data, split_pretrain_data, build_CP_vocab
from random import shuffle
from miditok.data_augmentation import augment_dataset
from cp_transformer import CPWordTransformer
from tqdm import tqdm
import logging
import os


def main():
    
    logging.basicConfig(
        filename='pretrain_log.log',
        level=logging.INFO,
        format='%(asctime)s — %(levelname)s — %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    os.makedirs("pretrain_checkpoints", exist_ok=True)

    tokenizer = CPWord()

    # download and split pretrain data only if the folder does not exist
    if not os.path.exists("midis"):
        pretrain_file_id = "1BDEPaEWFEB2ADquS1VYp5iLZYVngw799"
        url = f"https://drive.google.com/uc?id={pretrain_file_id}"
        
        load_pretrain_data(url, "midis.zip", "midis")
        midis_path = list(Path("midis/midis").resolve().glob("**/*.mid"))
        if not os.path.exists("compound2id.pkl"):
            compound2id, id2compound = build_CP_vocab(midis_path, tokenizer)
        
        split_pretrain_data("midis", tokenizer, 1024)
        
    midi_paths = list(Path("pretrain_data/dataset_train").resolve().glob("**/*.mid"))

    dataset = DatasetMIDI(
        files_paths=midi_paths,
        tokenizer=tokenizer,
        max_seq_len=1024,
        bos_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer["BOS_None"],
    )
    
    collator = DataCollator(tokenizer.pad_token_id)
    data_loader = DataLoader(dataset=dataset, collate_fn=collator, batch_size=16)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CPWordTransformer(len(compound2id.items()), tokenizer)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)

    pad_tuple = tuple(tokenizer.pad_token_id for _ in range(len(tokenizer.vocab)))
    compound_pad_id = compound2id[pad_tuple]
    criterion = nn.CrossEntropyLoss(ignore_index=compound_pad_id)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # ==== Resume checkpoint ====
    checkpoint_path = "pretrain_checkpoints/decoder_epoch_4.pt"  # change if needed
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        logging.info(f"Resumed training from checkpoint: epoch {checkpoint['epoch']}")
        
        
    num_epochs = 20
    save_every = 2

    model.train()
    step = 0
    log_interval = 500
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for _, batch in enumerate(tqdm(data_loader)):
            step += 1

            input_ids = batch['input_ids'].to(device)            #  (B, T, f)
            attention_mask = batch['attention_mask'].to(device)  # (B, T)

            decoder_input = input_ids[:, :-1]        # (B, T - 1, f)
            attn_mask = attention_mask[:, :-1]       # (B, T - 1)

            tgt = input_ids[:, 1:]                # (B, T - 1, f)
            tgt_key_padding_mask = (attn_mask == 0)

            output = model(
                decoder_input,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )  
            logits = output.reshape(-1, output.size(-1)) # (B*(T-1), cp_vocab_size)

            B, Tm1, f = tgt.shape
            flat = tgt.reshape(-1, f).tolist()
            flat_ids = [ compound2id[tuple(feat)] for feat in flat ]
            tgt_compound = torch.tensor(flat_ids,
                            device=tgt.device,
                            dtype=torch.long
                   ).view(B, Tm1)

            loss = criterion(logits, tgt_compound.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            if step % log_interval == 0:
                avg_train_loss = epoch_loss / step
                log_msg = f"step {step} - Loss: {avg_train_loss:.4f}"
                logging.info(log_msg)
            

        log_msg = f"Epoch {epoch+1} — Loss: {epoch_loss / len(data_loader):.4f}"
        print(log_msg) 
        logging.info(log_msg)

        if epoch % save_every == 0 and epoch != 0:
            checkpoint_path = f"pretrain_checkpoints/decoder_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")


    
if __name__ == "__main__":
    main()
