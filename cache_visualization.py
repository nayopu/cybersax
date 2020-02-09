"""PyAudio Example: Play a wave file."""

import matplotlib.pyplot as plt
import numpy as np
import threading
from utils import load
import pyaudio
import time
import wave

class Replay(object):
    def __init__(self, cache, samples_per_fft, frame_size, freq_step):
        self.i = 0
        self.cache = cache
        self.frame_size = frame_size
        self.freq_step = freq_step
        self.buf = np.zeros(samples_per_fft, dtype=np.float32)
        # plt.ion()
        # plt.show()

    def callback(self, in_data, frame_count, time_info, status):
        print(self.i)
        data = cache[self.i]

        # convert to numpy
        data_np = np.fromstring(data, np.int16)

        # Shift the buffer down and new data in
        self.buf[:-self.frame_size] = self.buf[self.frame_size:]
        self.buf[-self.frame_size:] = data_np

        # Run the FFT on the windowed buffer
        fft = np.fft.rfft(self.buf * window)
        fft_plot = np.abs(fft[imin:imax])
        freqs = np.arange(imin, imax) * self.freq_step
        plt.subplot(211)
        plt.plot(np.arange(len(self.buf)), self.buf)
        plt.subplot(212)
        plt.plot(freqs, fft_plot)
        plt.draw()
        plt.pause(0.01)
        plt.clf()

        self.i += 1
        return (data, pyaudio.paContinue)

NOTE_MIN = 10       # C4
NOTE_MAX = 69       # A4
FSAMP = 48000       # Sampling frequency in Hz
FRAME_SIZE = 2048   # How many samples per frame?
FRAMES_PER_FFT = 32 # FFT takes average across how many frames?
DEVICE_NAME_PLAY = 'USB Audio Device: - (hw:3,0)'

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
# Get min/max index within FFT of notes we care about.
# See docs for numpy.rfftfreq()
def note_to_fftbin(n): return number_to_freq(n)/FREQ_STEP
imin = max(0, int(np.floor(note_to_fftbin(NOTE_MIN-1))))
imax = min(SAMPLES_PER_FFT, int(np.ceil(note_to_fftbin(NOTE_MAX+1))))

# Create Hanning window function
window = 0.5 * (1 - np.cos(np.linspace(0, 2*np.pi, SAMPLES_PER_FFT, False)))

# Print initial text
print('sampling at', FSAMP, 'Hz with max resolution of', FREQ_STEP, 'Hz')

p = pyaudio.PyAudio()

cache = load('test.pkl')
print(f'Cache length is {len(cache)}')

wavFile = wave.open('test.wav', 'wb')
wavFile.setnchannels(1)
wavFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
wavFile.setframerate(FSAMP)
wavFile.writeframes(b"".join(cache)) #Python3用

wavFile.close()

replay = Replay(cache, SAMPLES_PER_FFT, FRAME_SIZE, FREQ_STEP)

device_index = {}
for i in range(p.get_device_count()):
    device_index[p.get_device_info_by_index(i)["name"]] = i
    print(p.get_device_info_by_index(i))

stream_white_noise = pyaudio.PyAudio().open(format=pyaudio.paInt16,
                                channels=1,
                                rate=FSAMP,
                                frames_per_buffer=FRAME_SIZE,
                                output=True,
                                output_device_index=device_index[DEVICE_NAME_PLAY],
                                stream_callback=replay.callback)

stream_white_noise.start_stream()

while True:
    try:
        time.sleep(0.1)
    except KeyboardInterrupt:
        break