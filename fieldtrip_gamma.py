"""
Reproduce FieldTrip example here:
http://www.fieldtriptoolbox.org/tutorial/beamformingextended/
"""

# Authors: Mainak Jas <mainakjas@gmail.com>

import numpy as np

from mne import create_info, EpochsArray
from mne.io import read_epochs_fieldtrip
from mne.channels import read_layout

from mne.externals.pymatreader.pymatreader import read_mat

sfreq = 400.

ft_struct = read_mat('subjectK.mat',
                   ignore_fields=['previous'],
                   variable_names=['data_left', 'data_right'])


# get epochs data
data = ft_struct['data_left']['trial']
ch_names = ft_struct['data_left']['label']

n_trials = len(data)
max_n_times = max([d.shape[1] for d in data])
n_channels = len(ch_names)

data_epochs = np.empty((n_trials, n_channels, max_n_times))

for idx, d in enumerate(data):
      n_times = data[idx].shape[1]
      data_epochs[idx, :, :n_times] = data[idx]

# get channel types and create info
ch_types = ['grad'] * len(ch_names)
ch_types[ch_names.index('EMGlft')] = 'emg'
ch_types[ch_names.index('EMGrgt')] = 'emg'

info = create_info(ch_names, sfreq, ch_types=ch_types)

epochs = EpochsArray(data_epochs, info)
epochs.crop(1.1, 1.9)
epochs.plot(scalings=dict(grad=10e-13), n_epochs=5, n_channels=10)

epochs.plot_psd()

layout = read_layout('ctf151')
# XXX: what is normalize = True? in the MNE docstring of this function
# TODO: contrast the PSD with baseline
epochs.plot_psd_topomap(bands=[(40, 70, 'Gamma')], layout=layout, outlines='skirt')


sdfdfdf
import matplotlib.pyplot as plt
from mne.viz.utils import center_cmap
from mne.time_frequency import tfr_multitaper

freqs = np.arange(40, 70, 1)  # frequencies from 2-35Hz
n_cycles = 30  # use constant t/f resolution
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
baseline = [0, 0.8]  # baseline interval (in s)
cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white
kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
              buffer_size=None)  # for cluster test

# Run TF decomposition overall epochs
tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                     use_fft=True, return_itc=False, average=False,
                     decim=1)
tfr.crop(0, 1.9)
tfr.apply_baseline(baseline, mode="percent")
