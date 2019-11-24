"""Gamma tutorial of FieldTrip in source space.

First download:
ftp://ftp.fieldtriptoolbox.org/pub/fieldtrip/tutorial/sensor_analysis/subjectK.mat
"""

# Authors: Mainak Jas <mainakjas@gmail.com>

import os.path as op

import numpy as np

import mne
from mne.datasets import sample
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import csd_morlet
from mne.datasets import testing
from mne.io import read_raw_ctf

from utils import load_data

ctf_dir = op.join(testing.data_path(download=False), 'CTF')
ctf_fname_catch = 'catch-alp-good-f.ds'

raw = read_raw_ctf(op.join(ctf_dir, ctf_fname_catch), clean_names=True,
                   preload=True)
raw.pick_types(meg=True, ref_meg=False)

####################
# READ DATA

sfreq = 400.
epochs = load_data('subjectK.mat', raw.info.copy(), sfreq=sfreq, tmin=-1.,
                   data_name='data_left')

fs_dir = op.join(op.dirname(sample.data_path()), 'MNE-fsaverage-data',
                 'fsaverage')
subjects_dir = op.dirname(fs_dir)

####################
# SOURCE LOCALIZATION

subject = 'fsaverage'
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
fwd = mne.make_forward_solution(epochs.info, trans=None, src=src,
                                bem=bem, eeg=False, mindist=5.0, n_jobs=1,
                                ignore_ref=True)

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
