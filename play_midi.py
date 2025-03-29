#! /usr/bin/env python3

"""
This script plays a MIDI file using the pygame library.
Adopted from: https://www.daniweb.com/programming/software-development/code/216979/embed-and-play-midi-music-in-your-code-python

Usage:
    python play_midi.py <midi_file>
"""

import pygame
import argparse
import sys

def play_music(music_file):
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
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        # check if playback has finished
        clock.tick(30)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Play a MIDI file using pygame')
    parser.add_argument('midi_file', help='Path to the MIDI file to play')
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
        # Play the MIDI file
        play_music(args.midi_file)
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()