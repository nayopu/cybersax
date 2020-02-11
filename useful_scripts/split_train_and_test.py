import os
import wave

from pydub import AudioSegment
import pyaudio

SRC_DIR = 'processed/20200208_wav'
DST_DIR = 'processed/20200208_wav_eval'
LEN_TRAIN = 10000

if __name__ == '__main__':
    names = os.listdir(SRC_DIR)
    src_fps = [f'{SRC_DIR}/{x}' for x in names]
    dir_tr = os.path.join(DST_DIR, 'train')
    dir_te = os.path.join(DST_DIR, 'test')
    os.makedirs(dir_tr, exist_ok=True)
    os.makedirs(dir_te, exist_ok=True)
    for name, src_fp in zip(names, src_fps):
        sound = AudioSegment.from_file(src_fp, "wav")
        sound_tr = sound[:LEN_TRAIN]
        sound_te = sound[LEN_TRAIN:]

        sound_tr.export(f'{dir_tr}/{name.split(".")[0]}.wav', "wav")
        sound_te.export(f'{dir_te}/{name.split(".")[0]}.wav', "wav")

        print(f'file: {name}, train.len: {sound_tr.__len__()}, test.len: {sound_te.__len__()}')
