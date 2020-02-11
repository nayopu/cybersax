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

FRAME_SIZE = 1024
DATA_DIR = 'exp_data/20200208'
RESULT_DIR = 'exp_data/20200208'

# NAMES = ['D3',
#          'D4',
#          'G3',
#          'G4',
#          'B3',
#          'B4']
TEST_RATE = 0.1
SEED = 0
SCALE = 1e10
FSAMP = 48000
FREQ_STEP = float(FSAMP)/FRAME_SIZE
# file_names = [f'{DIR}/{x}.pkl' for x in NAMES]
file_names = glob(f'{DATA_DIR}/*3.pkl')

names = [os.path.basename(x).split('.')[0] for x in file_names]
nclasses = len(names)
caches = [load(x) for x in file_names]
appended_caches = [b"".join(x) for x in caches]
amps = [np.fromstring(x, np.int16) for x in appended_caches]
lens = [len(x) for x in amps]
len_min = np.min(lens)
reduced_amps = [x[:len_min] for x in amps]
reshaped_amps = [x.reshape([-1, FRAME_SIZE]) for x in reduced_amps]

# Create Hanning window function
window = 0.5 * (1 - np.cos(np.linspace(0, 2*np.pi, FRAME_SIZE, False)))

X = []
y = []
for class_, reshaped_amp in zip(range(nclasses), reshaped_amps):
    for amp in reshaped_amp:
        spectrum = np.abs(np.fft.rfft(amp * window).real)
        # spectrum = np.abs(np.fft.rfft(amp * window).real)[: int(400 // FREQ_STEP)]
        X.append(spectrum)
        y.append(class_)
    #
    freqs = np.arange(len(spectrum)) * FREQ_STEP
    plt.subplot(211)
    plt.plot(amp)
    plt.subplot(212)
    plt.plot(freqs, spectrum)
    plt.draw()
    # plt.savefig('figure.png')
    plt.show()
#
X, y = np.asarray(X), np.asarray(y)
nsamples = X.shape[0]

# scale
X = X * SCALE

# permutate
np.random.seed(0)
perm_indices = np.random.permutation(np.arange(nsamples))
X, y = X[perm_indices], y[perm_indices]

nsamp_test = int(nsamples * TEST_RATE)
trX, trY = X[:-nsamp_test], y[:-nsamp_test]
teX, teY = X[-nsamp_test:], y[-nsamp_test:]

clf = LogisticRegression(max_iter=1000).fit(trX, trY)
# clf = MLPClassifier().fit(trX, trY)
score = clf.score(teX, teY)

disp = plot_confusion_matrix(clf, teX, teY,
                                 display_labels=names,
                                 cmap=plt.cm.Blues)

# disp.figure_.sa
print(f'Mean accuracy {score}')

# save
os.makedirs(RESULT_DIR, exist_ok=True)
disp.figure_.savefig(f'{RESULT_DIR}/confusion.png')
dump(f'{RESULT_DIR}/clf.pkl', clf)
