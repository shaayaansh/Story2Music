from miditok import REMI, TokenizerConfig
from miditok.utils import split_files_for_training
from pathlib import Path
import os
import muspy
from random import shuffle
from miditok.utils import split_files_for_training
import torch
import gdown
import zipfile
import pickle


def convert_to_midi(token_ids, tokenizer, dump_path):
    """
    this function converts token_ids to actual tokens
    then the actual tokens are dumped into a midi file
    """
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    actual_tokens = [[id_to_token[i] for i in token_ids]]
    
    with open("generated_tokens.txt", "w") as f:
        for token in actual_tokens[0]:
            f.write(token + "\n")
            
    generated_midi = tokenizer(actual_tokens)
    generated_midi.dump_midi(dump_path)
    
    
    
def get_eval_metrics(midi_path):
    music = muspy.read_midi(midi_path)
    # get metrics
    pitch_range = muspy.pitch_range(music)
    n_pitch = muspy.n_pitches_used(music)
    n_pitch_class = muspy.n_pitch_classes_used(music)
    polyphony = py.polyphony(music)
    polyphony_rate = muspy.polyphony_rate(music)
    empty_beat_rate = muspy.empty_beat_rate(music)
    
    return [
        pitch_range,
        n_pitch,
        n_pitch_class,
        polyphony,
        polyphony_rate,
        empty_beat_rate,
    ]
    

def generate_causal_mask(size):
    return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)


def load_pretrain_data(download_url, output_zip, extracted_file_name):
    """ downloads pretrain data and unzips it

    """

    print("downloading pretrain data: ")
    gdown.download(download_url, output_zip)
    
    print("unzipping pretrain data: ")
    with zipfile.ZipFile(output_zip, 'r') as zip_f:
        zip_f.extractall(extracted_file_name)

    print("pretrain data unzipped.")
    os.remove(output_zip)
    print("removed zip file")

    return extracted_file_name


def split_pretrain_data(midi_path, tokenizer, max_len=1024):
    """ splits the pretrain data into train/val/test
    """
    
    midi_paths = list(Path(midi_path).resolve().glob('**/*.mid'))
    total_num_files = len(midi_paths)
    num_files_train = int(total_num_files * 0.8)
    num_files_valid = int(total_num_files * 0.1)

    midi_paths_train = midi_paths[:num_files_train]
    midi_paths_valid = midi_paths[num_files_train:num_files_train + \
        num_files_valid]
    midi_paths_test = midi_paths[num_files_train+num_files_valid:]

    for files_path, subset_name in (
        (midi_paths_train, "train"),
        (midi_paths_valid, "validation"),
        (midi_paths_test, "test")
    ):
        subset_chunks_dir = Path(f"pretrain_data/dataset_{subset_name}")
        split_files_for_training(
            files_paths=files_path,
            tokenizer=tokenizer,
            save_dir=subset_chunks_dir,
            max_seq_len=max_len,
            num_overlap_bars=2,
        )


def build_CP_vocab(midis_path, tokenizer):
    """
    builds a dictionary for all compound tokens seen in the data
    args:
    midis_path: Path --> path to the main folder holding midi files
    tokenizer --> the tokenizer used for tokenizing data
    Return:
    compound2id: Dictionary --> dictionary of all compound tokens to their IDs
    id2compound: Dictionary --> inverse dictionary of compound2id
    """

    compound_set = set()

    for path in midis_path:
        out = tokenizer.encode(path)
        seqs = out if isinstance(out, list) else [out]
        for seq in seqs:
            for feat_ids in seq.ids:
                compound_set.add(tuple(feat_ids))

    compound2id = { feat_tuple: idx
                    for idx, feat_tuple in enumerate(sorted(compound_set)) }
    compound_vocab_size = len(compound2id)

    print("Found", compound_vocab_size, "unique compound tokens")

    n_streams = len(tokenizer.vocab)
    # build the special tuples (BOS, EOS, PAD)
    bos_tuple = tuple(tokenizer.vocab[f]["BOS_None"] for f in range(n_streams))
    eos_tuple = tuple(tokenizer.vocab[f]["EOS_None"] for f in range(n_streams))
    pad_id   = tokenizer.pad_token_id
    pad_tuple= tuple(pad_id for _ in range(n_streams))

    # build the inverse mapping
    id2compound = {idx: tok for tok, idx in compound2id.items()}

    for special, name in [(bos_tuple, "BOS"), (eos_tuple, "EOS"), (pad_tuple, "PAD")]:
        if special not in compound2id:
            new_id = len(compound2id)
            compound2id[special] = new_id

            id2compound[new_id] = "+".join(
                # find the token string back from the dict:
                next(tok for tok, idx in tokenizer.vocab[f].items() if idx == special[f])
                for f in range(n_streams)
            )

    with open("compound2id.pkl", "wb") as f:
        pickle.dump(compound2id, f)

    with open("id2compound.pkl", "wb") as f:
        pickle.dump(id2compound, f)
    
    return compound2id, id2compound

