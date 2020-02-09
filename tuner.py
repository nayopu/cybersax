#! /usr/bin/env python
######################################################################
# tuner.py - a minimal command-line guitar/ukulele tuner in Python.
# Requires numpy and pyaudio.
######################################################################
# Author:  Matt Zucker
# Date:    July 2016
# License: Creative Commons Attribution-ShareAlike 3.0
#          https://creativecommons.org/licenses/by-sa/3.0/us/
######################################################################

import os
import sys
import time
import numpy as np
import pyaudio
import wave
from utils import dump
######################################################################
# Feel free to play with these numbers. Might want to change NOTE_MIN
# and NOTE_MAX especially for guitar/bass. Probably want to keep
# FRAME_SIZE and FRAMES_PER_FFT to be powers of two.

time.sleep(2.)

NOTE_MIN = 60       # C4
NOTE_MAX = 69       # A4
# FSAMP = 44100       # Sampling frequency in Hz
FSAMP = 48000       # Sampling frequency in Hz
FRAME_SIZE = 2048   # How many samples per frame?
FRAMES_PER_FFT = 32 # FFT takes average across how many frames?
DEVICE_NAME_PLAY = 'USB Audio Device: - (hw:3,0)'
DEVICE_NAME_REC = 'USB Audio Device: - (hw:2,0)'
# WAV_FILE = 'whitenoise_1k_12db.wav'
WAV_FILE = 'whitenoise.wav'
# WAV_FILE = sys.argv[1]
NAME = 'F#5'
DURATION = 20.
PREPARE_DURATION = 3
######################################################################
# Derived quantities from constants above. Note that as
# SAMPLES_PER_FFT goes up, the frequency step size decreases (so
# resolution increases); however, it will incur more delay to process
# new sounds.

SAMPLES_PER_FFT = FRAME_SIZE*FRAMES_PER_FFT
FREQ_STEP = float(FSAMP)/SAMPLES_PER_FFT

######################################################################
# For printing out notes

NOTE_NAMES = 'C C# D D# E F F# G G# A A# B'.split()

######################################################################
# These three functions are based upon this very useful webpage:
# https://newt.phys.unsw.edu.au/jw/notes.html

def freq_to_number(f): return 69 + 12*np.log2(f/440.0)
def number_to_freq(n): return 440 * 2.0**((n-69)/12.0)
def note_name(n): return NOTE_NAMES[n % 12] + str(n/12 - 1)

######################################################################
# Ok, ready to go now.

# Get min/max index within FFT of notes we care about.
# See docs for numpy.rfftfreq()
def note_to_fftbin(n): return number_to_freq(n)/FREQ_STEP
imin = max(0, int(np.floor(note_to_fftbin(NOTE_MIN-1))))
imax = min(SAMPLES_PER_FFT, int(np.ceil(note_to_fftbin(NOTE_MAX+1))))

# Allocate space to run an FFT.
buf = np.zeros(SAMPLES_PER_FFT, dtype=np.float32)
num_frames = 0

# get device indice
p = pyaudio.PyAudio()
device_index = {}
for i in range(p.get_device_count()):
    device_index[p.get_device_info_by_index(i)["name"]] = i
    print(p.get_device_info_by_index(i))

# play white noise
wf = wave.open(WAV_FILE, 'rb')
def callback_play(in_data, frame_count, time_info, status):
    data = wf.readframes(frame_count)
    # print(frame_count, len(data), status)
    if len(data) < 2 * frame_count:
        print(len(data))
        wf.rewind()
        data = wf.readframes(frame_count)
        # print('rewinded')
    return (data, pyaudio.paContinue)

# Create Hanning window function
window = 0.5 * (1 - np.cos(np.linspace(0, 2*np.pi, SAMPLES_PER_FFT, False)))

# Print initial text
print('sampling at', FSAMP, 'Hz with max resolution of', FREQ_STEP, 'Hz')

# start noise playing
stream_white_noise = pyaudio.PyAudio().open(format=pyaudio.paInt16,
                                channels=1,
                                rate=FSAMP,
                                frames_per_buffer=FRAME_SIZE,
                                output=True,
                                output_device_index=device_index[DEVICE_NAME_PLAY],
                                stream_callback=callback_play)
stream_white_noise.start_stream()

cache = []

# Finger prepare
for i in range(PREPARE_DURATION):
    print("Recording starts in {} sec.".format(PREPARE_DURATION-i))
    time.sleep(1)

# start recording
# Initialize audio
stream = p.open(format=pyaudio.paInt16,
                                channels=1,
                                rate=FSAMP,
                                input=True,
                                frames_per_buffer=FRAME_SIZE,
                                input_device_index=device_index[DEVICE_NAME_REC])
stream.start_stream()

print('Start recording ...')
start_time = time.time()
# As long as we are getting data:
while time.time() - start_time < DURATION:
    try:
        data_string = stream.read(FRAME_SIZE)
        data_np = np.fromstring(data_string, np.int16)

        # Shift the buffer down and new data in
        buf[:-FRAME_SIZE] = buf[FRAME_SIZE:]
        buf[-FRAME_SIZE:] = data_np

        # Run the FFT on the windowed buffer
        fft = np.fft.rfft(buf * window)

        # Get frequency of maximum response in range
        freq = (np.abs(fft[imin:imax]).argmax() + imin) * FREQ_STEP

        # Get note number and nearest note
        n = freq_to_number(freq)
        n0 = int(round(n))

        # Console output once we have a full buffer
        num_frames += 1

        if num_frames >= FRAMES_PER_FFT:
            print('freq: {:7.2f} Hz     note: {:>3s} {:+.2f}'.format(
                freq, note_name(n0), n-n0))

        cache.append(data_string)

    except KeyboardInterrupt:
        # if not os.path.exists(f'{NAME}.pkl'):
        #     dump(f'{NAME}.pkl', cache)
        #     print('Pickled cache was dumped "{NAME}".')
        break

if not os.path.exists(f'{NAME}.pkl'):
    dump(f'{NAME}.pkl', cache)
    print(f'Pickled cache was dumped "{NAME}".')
else:
    print(f'{NAME} already exists.')