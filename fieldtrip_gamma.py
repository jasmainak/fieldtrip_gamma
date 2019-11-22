"""
Reproduce FieldTrip example here:
http://www.fieldtriptoolbox.org/tutorial/beamformingextended/
"""

# Authors: Mainak Jas <mainakjas@gmail.com>

import numpy as np

from mne import create_info, EpochsArray
from mne.filter import notch_filter
from mne.channels import read_layout

from mne.time_frequency import tfr_multitaper

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

from mne.io.fieldtrip.utils import _create_montage
montage = _create_montage(ft_struct['data_left'])

data_epochs = notch_filter(data_epochs, sfreq, [50., 100., 150.])
epochs = EpochsArray(data_epochs, info, tmin=-1.)
for ch_name, dig in zip(montage.ch_names, montage.dig):
    idx = epochs.ch_names.index(ch_name)
    epochs.info['chs'][idx]['loc'] = np.zeros((12,))
    epochs.info['chs'][idx]['loc'][:3] = dig['r']

epochs.crop(-1., 1.3)
epochs.plot(scalings=dict(grad=10e-13), n_epochs=5, n_channels=10)

epochs.plot_psd()

freqs = np.arange(20., 100., 1.)
n_cycles = 44
time_bandwidth = 4.0  # Least possible frequency-smoothing (1 taper)

# XXX: how to deal with unequal trials?
power = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                       time_bandwidth=time_bandwidth, return_itc=False,
                       average=True)

layout = read_layout('CTF151.lay')

# Plot results.
vmin, vmax = -0.45, 0.6
picks = ['MRP31', 'MZP02', 'MRO12', 'MR021', 'ML012']
# XXX: label is incorrect
baseline = (-0.8, 0.)
mode = 'percent'
tmax = 1.1
power.plot(picks, baseline=baseline, mode=mode, vmin=vmin, vmax=vmax,
           layout=layout, tmin=-0.8, tmax=tmax)
power.plot_topomap(layout=layout, fmin=40., fmax=70., tmin=0.3, tmax=tmax,
                   baseline=baseline, mode=mode, show_names=False,
                   outlines='skirt')

# Download fsaverage files
import mne
import os.path as op

from mne.datasets import sample

fs_dir = op.join(op.dirname(sample.data_path()), 'MNE-fsaverage-data',
                 'fsaverage')
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
