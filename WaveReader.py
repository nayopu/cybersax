# -*- coding: utf-8 -*-
import numpy as np
import wave
import pyaudio
import time

class WaveReader:

    def __init__(self, fname, chunk_size, out_device_index=None):
        self.wav = wave.open(fname, "rb")
        self.num_channels = self.wav.getnchannels()
        self.sample_width = self.wav.getsampwidth()
        self.framerate = self.wav.getframerate()
        self.num_frames = self.wav.getnframes()
        self.chunk_size = chunk_size
        self.amp = (2**8) ** self.sample_width / 2
        type_map = {1:'int8', 2:'int16', 4:'int32'}
        self.sample_dtype = type_map[self.sample_width]
        self.chunk_len = self.chunk_size * self.num_channels * self.sample_width
        self.last_frames = np.zeros(self.chunk_size * self.num_channels)
        self.pa_obj = pyaudio.PyAudio()
        self.stream = self.pa_obj.open(
            format=self.pa_obj.get_format_from_width(self.sample_width),
            channels=self.num_channels,
            rate=self.framerate,
            output=True,
            output_device_index=out_device_index
        )
        self.finished = False
        self.last_chunk = None

    def __del__(self):
        self.close()

    def close(self):
        if self.finished == False:
            self.wav.close()
            self.stream.stop_stream()
            self.stream.close()
            self.pa_obj.terminate()
            self.finished = True

    def read(self):
        if self.finished == True:
            chunk = self.last_chunk
        else :
            chunk = self.wav.readframes(self.chunk_size)
            if len(chunk) < self.chunk_len:
                self.close()
                chunk = self.last_chunk
                # self.wav.rewind()
                # chunk = 0
        new_frames = np.frombuffer(chunk, self.sample_dtype) / self.amp
        frames = np.concatenate([self.last_frames, new_frames])
        self.last_frames = new_frames
        self.last_chunk = chunk
        return (chunk, frames)

    def write(self):
        if self.finished == False:
            self.stream.write(self.last_chunk)
