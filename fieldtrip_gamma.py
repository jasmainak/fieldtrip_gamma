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
                   variable_names=['data_left'])


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
epochs.plot(scalings=dict(grad=10e-13), n_epochs=5, n_channels=10)

layout = read_layout('ctf151')
epochs.plot_psd_topomap(bands=[(40, 70, 'Gamma')], layout=layout, outlines='skirt')
