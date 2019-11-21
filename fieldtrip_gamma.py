"""
Reproduce FieldTrip example here:
http://www.fieldtriptoolbox.org/tutorial/beamformingextended/
"""

# Authors: Mainak Jas <mainakjas@gmail.com>

import numpy as np

from mne import create_info, EpochsArray
from mne.channels import read_layout
from mne.filter import notch_filter

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

data_epochs = np.zeros((n_trials, n_channels, max_n_times))

for idx, d in enumerate(data):
    n_times = data[idx].shape[1]
    data_epochs[idx, :, :n_times] = data[idx]

# get channel types and create info
ch_types = ['grad'] * len(ch_names)
ch_types[ch_names.index('EMGlft')] = 'emg'
ch_types[ch_names.index('EMGrgt')] = 'emg'

info = create_info(ch_names, sfreq, ch_types=ch_types)

data_epochs = notch_filter(data_epochs, sfreq, [50., 100., 150.])
epochs = EpochsArray(data_epochs, info, tmin=-1.)
epochs.crop(-1., 1.3)
epochs.plot(scalings=dict(grad=10e-13), n_epochs=5, n_channels=10)

epochs.plot_psd()

from mne.time_frequency import psd_array_multitaper

epochs_baseline = epochs.copy().crop(-0.8, 0).get_data()
epochs_gamma = epochs.copy().crop(0.3, 1.1).get_data()

# XXX: psd_array_multitaper gives unequal freqs
psd_gamma, freqs1 = psd_array_multitaper(epochs_gamma, sfreq=epochs.info['sfreq'])
psd_baseline, freqs2 = psd_array_multitaper(epochs_baseline, sfreq=epochs.info['sfreq'])

psd_gamma = psd_gamma.mean(axis=0)
psd_baseline = psd_baseline.mean(axis=0)

# psd_gamma = psd_gamma[0]
# psd_baseline = psd_baseline[0]

freq_range = [40, 70]

idx1 = np.all(np.c_[freqs1 > freq_range[0], freqs1 < freq_range[1]], axis=1)
psd_gamma = np.sum(psd_gamma[:, idx1], axis=1)
idx2 = np.all(np.c_[freqs2 > freq_range[0], freqs2 < freq_range[1]], axis=1)
psd_baseline = np.sum(psd_baseline[:, idx2], axis=1)

# see https://github.com/mne-tools/mne-python/blob/master/mne/baseline.py
psd_norm = psd_gamma.copy()
psd_norm -= psd_baseline
psd_norm /= psd_baseline

from mne import EvokedArray

evoked = EvokedArray(psd_norm[:, None], epochs.info, tmin=0.)

layout = read_layout('CTF151.lay')
evoked.plot_topomap(layout=layout, times=[0.])

sdfdfddfdf

# XXX: what is normalize = True? in the MNE docstring of this function
# TODO: contrast the PSD with baseline

# XXX: epochs.filter doesn't work
# epochs.filter(40., 70.)

epochs.plot_psd_topomap(bands=[(40, 70, 'Gamma')], layout=layout,
                        outlines='skirt')
dfdfdf
from mne.time_frequency import tfr_morlet

freqs = np.arange(20, 100, 1)
n_cycles = freqs / 2.

power = tfr_morlet(epochs, freqs=freqs,
                   n_cycles=n_cycles, return_itc=False)
ch_average = ['MLO11', 'ML012', 'ML021', 'MLP31', 'MRO11', 'MRO12', 'MRO21',
              'MRO32', 'MRP31', 'MZO01', 'MZP02']
# XXX: label is incorrect
power.plot(ch_average, baseline=(-0.8, 0.), mode='percent',
           show=True, colorbar=False)


# Download fsaverage files
import mne
import os.path as op

fs_dir = '/autofs/space/meghnn_001/users/mjas/mne_data/MNE-fsaverage-data/fsaverage'
subjects_dir = op.dirname(fs_dir)

# Beamforming
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src,
                                bem=bem, eeg=False, mindist=5.0, n_jobs=1)

from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import csd_morlet

freqs = np.logspace(np.log10(40), np.log10(70), 20)

csd = csd_morlet(epochs, freqs, tmin=-0.8, tmax=1.1, decim=2)
csd_baseline = csd_morlet(epochs, freqs, tmin=-0.8, tmax=0, decim=2)
# ERS activity starts at 0.5 seconds after stimulus onset
csd_ers = csd_morlet(epochs, freqs, tmin=0.3, tmax=1.1, decim=2)

filters = make_dics(epochs.info, fwd, csd.mean(), pick_ori='max-power')

baseline_source_power, freqs = apply_dics_csd(csd_baseline.mean(), filters)
beta_source_power, freqs = apply_dics_csd(csd_ers.mean(), filters)

stc = beta_source_power / baseline_source_power
stc.subject = '01'  # it's mis-coded in fwd['src']
message = 'DICS source power in the 40-70 Hz frequency band'
brain = stc.plot(hemi='both', views='par', subjects_dir=subjects_dir,
                 subject=subject, time_label=message)
