"""PyAudio Example: Play a wave file."""
from glob import glob
import os

import matlab.engine

from ext_src.yin.audio_processing import audio_read
from ext_src.yin.yin import compute_yin

DATA_DIR = 'processed/20200208_wav_eval/test'
RESULT_DIR = 'result/20200208'
HOP_DUR = 1e-3
TRACEBACK_DIR = 3e-2
F_MIN = 130
F_MAX = 900
HARM_THRES = 0.85


data_fps = glob(os.path.join(DATA_DIR, '*'))

# yin
for data_fp in data_fps:

    sr, sig = audio_read(data_fp, formatsox=False)
    hop_size = int(sr * HOP_DUR)
    traceback_size = int(sr * TRACEBACK_DIR)
    pitches, harmonic_rates, argmins, times = compute_yin(sig, sr, None, traceback_size, hop_size, F_MIN, F_MAX, HARM_THRES)

    print('done')