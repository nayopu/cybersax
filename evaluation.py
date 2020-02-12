"""PyAudio Example: Play a wave file."""
from glob import glob
import os

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from modules.utils import *

YIN_CSV = 'result/csvs_130/yin_130.csv'
SWIPE_CSV = 'result/csvs_130/swipe_130.csv'
DIO_CSV = 'result/csvs_130/dio_130.csv'
LR_CSV = 'result/pred_csv/lr.csv'
LABEL_PKL = 'processed/20200208_wav_test_concat/gt_dict.pkl'
RESULT_DIR = 'result/scores_130'

W_DUR = 0.03
INTERVAL_DUR = 0.01
SAMP_RATE = 48000

wsize = SAMP_RATE * W_DUR
isize = SAMP_RATE * INTERVAL_DUR

df_yin = pd.read_csv(YIN_CSV, header=None)  # 32000
df_swipe = pd.read_csv(SWIPE_CSV, header=None)  # 32001
df_dio = pd.read_csv(DIO_CSV, header=None) # 320001
df_lr = pd.read_csv(LR_CSV, header=None)  # 31977

nsamples = len(df_yin)
print('nresults', ', '.join([f'{len(df)}' for df in [df_yin, df_swipe, df_dio, df_lr]]))

# prcess dataframes
## drop
df_swipe = df_swipe.drop(0)
df_dio = df_dio.drop(0)
## fill
df_yin = df_yin.fillna(method='ffill').fillna(method='bfill')
last_row = df_lr.index[-1]
for i in range(len(df_lr), nsamples):
    df_lr.loc[i] = df_lr.loc[last_row]

freqs_np = np.array(FREQS)
def get_nearest_tone_class(freq):
    class_ = np.argmin(np.abs(freqs_np - freq))
    return class_

# class prediction
pred_yin = df_yin[0].apply(get_nearest_tone_class).values
pred_swipe = df_swipe[0].apply(get_nearest_tone_class).values
pred_dio = df_dio[0].apply(get_nearest_tone_class).values
pred_lr = df_lr[0].values
pred_dict = {'yin': pred_yin, 'swipe': pred_swipe, 'dio': pred_dio, 'lr': pred_lr}

# ground truth
time_positions = load(LABEL_PKL)
gt = np.zeros(nsamples).astype(np.int)
for class_, tone in enumerate(TONES):
    start = int(time_positions[tone][0] / 1000 / INTERVAL_DUR)
    end = int(time_positions[tone][1] / 1000 / INTERVAL_DUR)
    gt[start:end] = class_
# confution matrix
os.makedirs(RESULT_DIR, exist_ok=True)
for k in pred_dict:
    cm = confusion_matrix(gt, pred_dict[k])
    disp = ConfusionMatrixDisplay(cm, TONES).plot()
    disp.figure_.set_size_inches(16., 14.)
    disp.figure_.savefig(f'{RESULT_DIR}/confusion_{k}.png')
    acc = accuracy_score(gt, pred_dict[k])
    print(f'Accuracy of {k}: {acc}')
