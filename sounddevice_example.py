#!/usr/bin/env python3
import math
import sounddevice as sd

sd.default.device = None
sd.default.samplerate = samplerate = 48000

duration = 1.5
volume = 0.3
frequency = 440

# fade time in seconds:
fade_in = 0.01
fade_out = 0.3

buffer = memoryview(bytearray(int(duration * samplerate) * 4)).cast('f')

for i in range(len(buffer)):
    buffer[i] = volume * math.cos(2 * math.pi * frequency * i / samplerate)

fade_in_samples = int(fade_in * samplerate)
for i in range(fade_in_samples):
    buffer[i] *= i / fade_in_samples

fade_out_samples = int(fade_out * samplerate)
for i in range(fade_out_samples):
    buffer[-(i + 1)] *= i / fade_out_samples

for mapping in ([1], [2], [1, 2]):
    sd.play(buffer, blocking=True, mapping=mapping)
    sd.sleep(500)