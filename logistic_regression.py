"""PyAudio Example: Play a wave file."""
from glob import glob
import numpy as np
import threading
from modules.utils import load
import time
import wave
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from modules.utils import *
import pandas as pd

TRAIN_DIR = 'processed/20200208_wav_eval/train'
TEST_PATH = 'processed/20200208_wav_test_concat/test.wav'
RESULT_DIR = 'result/pred_csv'
W_DUR = 0.03
INTERVAL_DUR = 0.01

tone_dict = {k: i for i, k in enumerate(TONES)}

def get_X(fp):
    with wave.open(fp, 'r') as f:
        fs = f.getframerate()
        frames = f.readframes(f.getnframes())
    x_all = np.frombuffer(frames, dtype="int16")
    size_all = len(x_all)
    wsize = int(fs * W_DUR)
    isize = int(fs * INTERVAL_DUR)
    nsamples = (size_all - wsize) // isize
    window = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, wsize, False)))
    # f_step = fs / wsize
    X = []
    for i in range(nsamples):
        x = x_all[i*isize:i*isize+wsize]
        spectrum = np.abs(np.fft.rfft(x * window).real)
        # spectrum = np.abs(np.fft.rfft(amp * window).real)[: int(400 // FREQ_STEP)]
        X.append(spectrum)
    print(f'File: {os.path.basename(fp)}, Used: {i*isize+wsize}/{size_all}')

    return X

file_names = []
for tone in TONES:
    file_names.append(f'{TRAIN_DIR}/{tone}.wav')
note_names = [os.path.basename(f).split('.')[0] for f in file_names]

trX = []
trY = []
for class_, name in enumerate(file_names):
    X_tmp = get_X(name)
    trX.extend(X_tmp)
    trY.extend([class_] * len(X_tmp))

trX, trY = np.asarray(trX), np.asarray(trY)
nsamples = trX.shape[0]

# permutate
np.random.seed(0)
perm_indices = np.random.permutation(np.arange(nsamples))
trX, trY = trX[perm_indices], trY[perm_indices]

# test
teX = np.asarray(get_X(TEST_PATH))

print(f'Dataset info: ntrains={len(trX)}, ntests={len(teX)}')
clf = LogisticRegression(max_iter=1e5).fit(trX, trY)
# clf = MLPClassifier().fit(trX, trY)
# score = clf.score(teX, teY)
pred = clf.predict(teX)

# disp.figure_.sa
# print(f'Mean accuracy {score}')

# save
os.makedirs(RESULT_DIR, exist_ok=True)
pd.DataFrame(pred).to_csv(f'{RESULT_DIR}/lr.csv', header=None, index=False)
dump(f'{RESULT_DIR}/note.pkl', note_names)
# disp.figure_.savefig(f'{RESULT_DIR}/confusion.png')
# dump(f'{RESULT_DIR}/clf.pkl', clf)

# NAMES = ['D3',
#          'D4',
#          'G3',
#          'G4',
#          'B3',
#          'B4']
# TEST_RATE = 0.1
# SEED = 0
# SCALE = 1e10
# FSAMP = 48000
# FREQ_STEP = float(FSAMP)/FRAME_SIZE
# # file_names = [f'{DIR}/{x}.pkl' for x in NAMES]
# file_names = glob(f'{DATA_DIR}/*3.pkl')
#
# names = [os.path.basename(x).split('.')[0] for x in file_names]
# nclasses = len(names)
# caches = [load(x) for x in file_names]
# appended_caches = [b"".join(x) for x in caches]
# amps = [np.fromstring(x, np.int16) for x in appended_caches]
# lens = [len(x) for x in amps]
# len_min = np.min(lens)
# reduced_amps = [x[:len_min] for x in amps]
# reshaped_amps = [x.reshape([-1, FRAME_SIZE]) for x in reduced_amps]
#
# # Create Hanning window function
#
# for class_, x_ in zip(range(nclasses), x):
#     for val in x_:
#
#     #
#     freqs = np.arange(len(spectrum)) * f_step
#     plt.subplot(211)
#     plt.plot(val)
#     plt.subplot(212)
#     plt.plot(freqs, spectrum)
#     plt.draw()
#     # plt.savefig('figure.png')
#     plt.show()
# #
# trX, trY = np.asarray(trX), np.asarray(trY)
# nsamples = trX.shape[0]
#
# # scale
# trX = trX * SCALE
#
# # permutate
# np.random.seed(0)
# perm_indices = np.random.permutation(np.arange(nsamples))
# trX, trY = trX[perm_indices], trY[perm_indices]
#
# nsamp_test = int(nsamples * TEST_RATE)
# trX, trY = trX[:-nsamp_test], trY[:-nsamp_test]
# teX, teY = trX[-nsamp_test:], trY[-nsamp_test:]
#
# clf = LogisticRegression(max_iter=1000).fit(trX, trY)
# # clf = MLPClassifier().fit(trX, trY)
# score = clf.score(teX, teY)
#
# disp = plot_confusion_matrix(clf, teX, teY,
#                                  display_labels=names,
#                                  cmap=plt.cm.Blues)
#
# # disp.figure_.sa
# print(f'Mean accuracy {score}')
#
# # save
# os.makedirs(RESULT_DIR, exist_ok=True)
# disp.figure_.savefig(f'{RESULT_DIR}/confusion.png')
# dump(f'{RESULT_DIR}/clf.pkl', clf)
