from miditok import REMI, TokenizerConfig
from pathlib import Path
import os
import muspy


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