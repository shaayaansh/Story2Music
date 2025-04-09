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



def load_and_split_pretraining_data(tokenizer):
    # Split the dataset into train/valid/test subsets, with 15% of the data for each of the two latter
    midi_paths = list(Path("pretrain_data", "midis").resolve().glob("**/*.mid"))

    total_num_files = len(midi_paths)
    num_files_valid = round(total_num_files * 0.15)
    num_files_test = round(total_num_files * 0.15)
    shuffle(midi_paths)
    midi_paths_valid = midi_paths[:num_files_valid]
    midi_paths_test = midi_paths[num_files_valid:num_files_valid + num_files_test]
    midi_paths_train = midi_paths[num_files_valid + num_files_test:]

    # Chunk MIDIs into sequences of length 1024
    for files_paths, subset_name in (
        (midi_paths_train, "train"), (midi_paths_valid, "valid"), (midi_paths_test, "test")
    ):

        # Split the MIDIs into chunks of sizes approximately about 1024 tokens
        subset_chunks_dir = Path(f"dataset_{subset_name}")
        split_files_for_training(
            files_paths=files_paths,
            tokenizer=tokenizer,
            save_dir=subset_chunks_dir,
            max_seq_len=1024,
            num_overlap_bars=2,
        )
        

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
    
    midi_paths = list(Path(midi_path).glob('**/*.mid'))
    total_num_files = len(midi_paths)
    num_files_train = total_num_files * 0.8
    num_files_valid = total_num_files * 0.1

    midi_paths_train = midi_paths[:total_files_train]
    midi_paths_valid = midi_paths[total_files_train:total_files_train + \
        total_files_valid]
    midi_paths_test = midi_paths[total_files_train+total_files_valid:]

    for files_path, subset_name in (
        (midi_path_train, "train"),
        (midi_path_valid, "validation"),
        (midi_path_test, "test")
    ):
        subset_chunks_dir = Path(f"pretrain_data/dataset_{subset_name}")
        subset_files_for_training(
            files_path=files_path,
            tokenizer=tokenizer,
            save_dir=subset_chunks_dir,
            max_seq_len=max_len,
            num_overlap_bars=2,
        )


