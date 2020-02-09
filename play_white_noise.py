"""PyAudio Example: Play a wave file."""

import pyaudio
import wave
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import threading

SAMP_CHUNK = 128
DEVICE_NAME_PLAY = 'USB Audio Device: - (hw:3,0)'
DEVICE_NAME_REC = 'USB Audio Device: - (hw:2,0)'
# WAV_FILE = 'whitenoise.wav'
WAV_FILE = '440Hz_44100Hz_16bit_30sec.wav'
FRAME_RATE = 44100
DEBUG = False

NOTE_MIN = 60       # C4
NOTE_MAX = 69       # A4
FSAMP = 22050       # Sampling frequency in Hz
FRAME_SIZE = 2048   # How many samples per frame?
FRAMES_PER_FFT = 16 # FFT takes average across how many frames?

mag = None

wf = wave.open(WAV_FILE, 'rb')

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

if DEBUG:
    def read():
        global mag
        data = wf.readframes(SAMP_CHUNK)
        mag = np.frombuffer(data, dtype="int16") / 32768.0
        time.sleep(1.)

    t = threading.Thread(target=read)
    t.start()

else:
    # get index
    device_index = {}
    for i in range(p.get_device_count()):
        device_index[p.get_device_info_by_index(i)["name"]] = i
        print(p.get_device_info_by_index(i))

    def callback_play(in_data, frame_count, time_info, status):
        # data_play = wf.readframes(frame_count)
        global mag
        mag =  np.frombuffer(in_data, dtype="int16")
        data_play = in_data
        return (data_play, pyaudio.paContinue)

    def callback(in_data, frame_count, time_info, status):
        global mag
        mag = np.frombuffer(in_data, dtype="int16") / 32768.0
        # ta_rec[:] = x
        # print(x.shape)
        # data_play = wf.readframes(frame_count)
        data_play = in_data
        return (data_play, pyaudio.paContinue)

    # open stream (2)
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=1,
                    rate=FRAME_RATE,
                    input=True,
                    output=True,
                    input_device_index=device_index[DEVICE_NAME_REC],
                    output_device_index=device_index[DEVICE_NAME_PLAY],
                    stream_callback=callback)

    # stream_play.start_stream()
    stream.start_stream()

# plt.figure(figsize=(15, 3))
# plt.ion()
# plt.show(block=False)
# plt.show()
# fig, ax = plt.subplots()

while True:
    if mag is not None:
        # print(np.sum(data_rec))
        # pass
        # in
        # data_in = stream_rec.read(SAMP_CHUNK, exception_on_overflow=False)
        # x = np.frombuffer(data_in, dtype="int16") / 32768.0
        #
        # plt.figure(figsize=(15, 3))
        # plt.plot(x)
        # plt.show()
        nsamples = len(mag)
        x = np.fft.fft(mag)
        freqs = np.arange(nsamples) * FRAME_RATE / nsamples
        x[freqs >= FRAME_RATE * 0.5] = 0
        mean_abs_mag = np.abs(x.real).mean()
        print('# samples: {}, Max Freqs: [{}], Mean of abs magnitude: {}'.format(nsamples, ','.join(['{:06.1f}'.format(freqs[x]) for x in np.argsort(-x.real)[:5]]), mean_abs_mag))
        # plt.plot(x.real[:int(len(x) / 2)])
        # plt.subplot(211)
        # plt.plot(np.arange(len(mag)), mag)
        # plt.subplot(212)
        # plt.psd(mag, len(mag), FRAME_RATE, sides='onesided')
        # plt.draw()
        # plt.pause(0.1)
        # plt.clf()
        # plt.figure(figsize=(15, 3))
        # plt.plot(mag.real[:int(len(mag) / 2)])
        # plt.show()
        time.sleep(.1)

# while stream_play.is_active():
#     time.sleep(0.1)
# read data
# data = wf.readframes(CHUNK)

# play stream (3)
# print('writing data ...')
# while True:
#     stream.write(data)
#     data = wf.readframes(CHUNK)
#     print(i)
#     i += 1
    # if len(data) < 2 * CHUNK:
    #     print(len(data))
    #     wf.rewind()

# stop stream (4)
# stream.stop_stream()
# stream.close()
# print('stream was closed.')

# close PyAudio (5)
# p.terminate()