import wave
import os

import pyaudio

from modules.utils import load

FSAMP = 48000       # Sampling frequency in Hz
SRC_DIR = 'exp_data/20200208'
DST_DIR = 'processed/20200208_wav'

p = pyaudio.PyAudio()

def pkl2wav(pkl_fp, wav_fp):
    pkl = load(pkl_fp)
    with wave.open(wav_fp, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        f.setframerate(FSAMP)
        f.writeframes(b"".join(pkl))

if __name__ == '__main__':
    names = os.listdir(SRC_DIR)
    src_fps = [f'{SRC_DIR}/{x}' for x in names]
    dst_fps = [f'{DST_DIR}/{x.split(".")[0]}.wav' for x in names]
    os.makedirs(DST_DIR)
    for src_fp, dst_fp in zip(src_fps, dst_fps):
        pkl2wav(src_fp, dst_fp)
        print(dst_fp)
