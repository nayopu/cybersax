import os
import wave

from pydub import AudioSegment
import pyaudio
import random
from modules.utils import dump

SEED = 0
SRC_DIR = 'processed/20200208_wav_eval/test'
DST_DIR = 'processed/20200208_wav_test_concat'
LEN_TEST = 10000

if __name__ == '__main__':
    names = os.listdir(SRC_DIR)
    names = random.sample(names, len(names))
    src_fps = [f'{SRC_DIR}/{x}' for x in names]
    os.makedirs(DST_DIR, exist_ok=True)
    sound_concat = None
    gt_dict = {}
    t_pre = 0
    for name, src_fp in zip(names, src_fps):
        sound = AudioSegment.from_file(src_fp, "wav")
        if sound_concat is None:
            sound_concat = sound
        else:
            sound_concat = sound_concat + sound
        gt_dict[name.split('.')[0]] = [t_pre, len(sound_concat)]
        t_pre = len(sound_concat)

    sound_concat.export(f'{DST_DIR}/test.wav', "wav")
    dump(f'{DST_DIR}/gt_dict.pkl', gt_dict)
    print(f'Generated wav: {DST_DIR}/test.wav')

