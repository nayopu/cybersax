# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import cupy as cp
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.ptime import time
import time

sys.path.append(os.path.dirname(__file__))
import WaveReader

CHUNK_SIZE = (2 ** 11)
MAX_PLOT_POWER = 50
OUT_INDEX = None

def fft_gpu(x_cpu):
    x_gpu = cp.asarray(x_cpu)
    y_gpu = cp.fft.rfft(x_gpu)
    y_cpu = cp.asnumpy(y_gpu)
    return abs(y_cpu)

def fft_cpu(x_cpu):
    y_cpu = np.fft.rfft(x_cpu)
    return abs(y_cpu)

args = sys.argv
if len(args) < 2:
    print('No input wave file specified.')
    sys.exit(-1)

reader = WaveReader.WaveReader(args[1], CHUNK_SIZE, OUT_INDEX)
print('Number of channles: %d' % (reader.num_channels))
print('Sample width      : %d' % (reader.sample_width))
print('Frame rate        : %d' % (reader.framerate))
print('Chunk size        : %d' % (reader.chunk_size))

nyquist_freq = reader.framerate / 2.0
resol = nyquist_freq / CHUNK_SIZE
x_data = np.arange(start=0.0, step=resol, stop=resol * (CHUNK_SIZE + 1))

app = QtGui.QApplication([])

p = pg.plot()
p.setTitle('Sound FFT Analyzer')
p.setRange(QtCore.QRectF(0, 0, int(resol * (CHUNK_SIZE + 1)), MAX_PLOT_POWER)) 
p.setLabel('bottom', 'Index', units='Hz')
p.showGrid(x=True, y=True)

curve_left = p.plot()
curve_right = p.plot()

fft_func = fft_cpu

def update():
    global x_data, curve_left, curve_right, reader, app, fft_func
    chunk, frames = reader.read()
    if reader.num_channels == 1:
        curve_left.setData(x_data, fft_func(frames), pen=(0, 0, 255))
    else:
        left = frames[0::reader.num_channels]
        right = frames[1::reader.num_channels]
        curve_left.setData(x_data, fft_func(left), pen=(0, 255, 0))
        curve_right.setData(x_data, fft_func(right), pen=(255, 0, 0))
    app.processEvents()  ## force complete redraw for every plot
    reader.write()

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()