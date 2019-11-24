"""Utility functions."""

# Authors: Mainak Jas <mainakjas@gmail.com>

import numpy as np
from mne.filter import notch_filter

from mne import EpochsArray
from mne.externals.pymatreader.pymatreader import read_mat


def load_data(fname, info, sfreq=400., tmin=-1., data_name='data_left'):
    ft_struct = read_mat(fname,
                         ignore_fields=['previous'],
                         variable_names=data_name)[data_name]

    # get epochs data
    data = ft_struct['trial']
    ch_names = ft_struct['label']

    n_trials = len(data)
    max_n_times = max([d.shape[1] for d in data])
    n_channels = len(ch_names) - 2

    data_epochs = np.zeros((n_trials, n_channels, max_n_times))

    for idx, d in enumerate(data):
        n_times = data[idx].shape[1]
        data_epochs[idx, :, :n_times] = data[idx][:-2, :]

    data_epochs = notch_filter(data_epochs, sfreq, [50., 100., 150.])
    info['sfreq'] = sfreq
    epochs = EpochsArray(data_epochs, info, tmin=-1.)

    return epochs
