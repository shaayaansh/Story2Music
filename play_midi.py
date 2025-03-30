#! /usr/bin/env python3

"""
This script plays a MIDI file using the pygame library and optionally saves the audio output.
Adopted from: https://www.daniweb.com/programming/software-development/code/216979/embed-and-play-midi-music-in-your-code-python

Usage:
    python play_midi.py <midi_file> [--output output.wav]
"""

import pygame
import argparse
import sys
import wave
import numpy as np
import time

def record_audio(duration, sample_rate=44100):
    """
    Record audio for the specified duration in seconds
    Returns the recorded audio data as a numpy array
    """
    pygame.mixer.init(sample_rate, -16, 2, 1024)
    pygame.sndarray.make_sound(np.zeros((1, 2), dtype=np.int16))
    
    # Create a buffer to store the audio data
    buffer = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Get the current audio data
        audio_data = pygame.sndarray.array(pygame.mixer.get_raw())
        buffer.append(audio_data)
    
    return np.concatenate(buffer)

def save_audio(audio_data, filename, sample_rate=44100):
    """
    Save the audio data as a WAV file
    """
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(2)  # stereo
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

def play_music(music_file, record=False, output_file=None):
    """
    stream music with mixer.music module in blocking manner
    this will stream the sound from disk while playing
    """
    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(music_file)
        print(f"Music file {music_file} loaded!")
    except pygame.error:
        print(f"File {music_file} not found! ({pygame.get_error()})")
        return

    if record:
        print("Recording started...")
        # Start recording before playing
        audio_data = record_audio(pygame.mixer.Sound(music_file).get_length())
        if output_file:
            save_audio(audio_data, output_file)
            print(f"Audio saved to {output_file}")
    else:
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            # check if playback has finished
            clock.tick(30)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Play a MIDI file using pygame and optionally save as audio')
    parser.add_argument('midi_file', help='Path to the MIDI file to play')
    parser.add_argument('--output', help='Output WAV file path (optional)')
    args = parser.parse_args()

    # Initialize pygame mixer
    freq = 44100    # audio CD quality
    bitsize = -16   # unsigned 16 bit
    channels = 2    # 1 is mono, 2 is stereo
    buffer = 1024   # number of samples
    pygame.mixer.init(freq, bitsize, channels, buffer)

    # Set volume (0 to 1.0)
    pygame.mixer.music.set_volume(0.8)

    try:
        # Play the MIDI file and optionally record
        play_music(args.midi_file, record=bool(args.output), output_file=args.output)
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()