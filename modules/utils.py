import pickle

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def dump(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

TONES = [
    'A#2',
    'B2',
    'C3',
    'C#3',
    'D3',
    'D#3',
    'E3',
    'F3',
    'F#3',
    'G3',
    'G#3',
    'A3',
    'A#3',
    'B3',
    'C4',
    'C#4',
    'D4',
    'D#4',
    'E4',
    'F4',
    'F#4',
    'G4',
    'G#4',
    'A4',
    'A#4',
    'B4',
    'C5',
    'C#5',
    'D5',
    'D#5',
    'E5',
    'F5'
]

FREQS = [
    138.59,  # 'A#2'
    146.83,  # 'B2'
    155.56,  # 'C3'
    164.81,  # 'C#3'
    174.61,  # 'D3'
    185.,  # 'D#3'
    196.,  # 'E3'
    207.65,  # 'F3'
    220.,  # 'F#3'
    233.08,  # 'G3'
    246.94,  # 'G#3'
    261.63,  # 'A3'
    277.18,  # 'A#3'
    293.66,  # 'B3'
    311.13,  # 'C4'
    329.63,  # 'C#4'
    349.23,  # 'D4'
    369.99,  # 'D#4'
    392.,  # 'E4'
    415.3,  # 'F4'
    440.,  # 'F#4'
    466.16,  # 'G4'
    493.88,  # 'G#4'
    523.25,  # 'A4'
    554.37,  # 'A#4'
    587.33,  # 'B4'
    622.25,  # 'C5'
    659.25,  # 'C#5'
    698.46,  # 'D5'
    739.99,  # 'D#5'
    783.99,  # 'E5'
    830.61  # 'F5'
]
