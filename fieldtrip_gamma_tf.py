"""
Reproduce FieldTrip example here:
http://www.fieldtriptoolbox.org/tutorial/beamformingextended/

First download:
ftp://ftp.fieldtriptoolbox.org/pub/fieldtrip/tutorial/sensor_analysis/subjectK.mat
"""

# Authors: Mainak Jas <mainakjas@gmail.com>

import numpy as np

from mne.channels import read_layout

from mne.time_frequency import tfr_multitaper
from mne.report import Report

from mne.io import read_raw_ctf

import os.path as op
from mne.datasets import testing

from utils import load_data

ctf_dir = op.join(testing.data_path(download=False), 'CTF')
ctf_fname_catch = 'catch-alp-good-f.ds'

raw = read_raw_ctf(op.join(ctf_dir, ctf_fname_catch), clean_names=True,
                   preload=True)
raw.pick_types(meg=True, ref_meg=False)

rep = Report()

####################
# READ DATA

sfreq = 400.
epochs = load_data('subjectK.mat', raw.info.copy(), sfreq=sfreq, tmin=-1.,
                   data_name='data_left')

epochs.crop(-1., 1.3)
epochs.plot(scalings=dict(grad=10e-13), n_epochs=5, n_channels=10)
epochs.plot_psd()

####################
# TIME-FREQUENCY


def iter_epochs(epochs, average=True):
    if average:
        return epochs
    for idx in range(len(epochs)):
        yield epochs[idx]


freqs = np.arange(20., 100., 1.)
n_cycles = 44
time_bandwidth = 4.0  # Least possible frequency-smoothing (1 taper)

layout = read_layout('CTF151.lay')

# Plot results.
# vmin, vmax = -0.45, 0.6
vmin, vmax = -2, 3
picks = ['MRP31', 'MZP02', 'MRO12', 'MR021', 'ML012']
# XXX: label is incorrect
baseline = (-0.8, 0.)
mode = 'percent'
tmax = 1.1
average = False

n_epochs = 140
# XXX: how to deal with unequal trials?
for idx, epoch in enumerate(iter_epochs(epochs[:n_epochs], average=average)):
    # XXX: This function needs to be more verbose. Say
    # when you process each epoch.
    print('[%d/%d]' % (idx, len(epochs)))
    power = tfr_multitaper(epoch, freqs=freqs, n_cycles=n_cycles,
                           time_bandwidth=time_bandwidth, return_itc=False,
                           average=True)
    fig1 = power.plot(picks, baseline=baseline, mode=mode, vmin=vmin,
                      vmax=vmax, layout=layout, tmin=-0.8, tmax=tmax)
    fig2 = power.plot_topomap(layout=layout, fmin=40., fmax=70.,
                              tmin=0.3, tmax=tmax,
                              baseline=baseline, mode=mode, show_names=False,
                              outlines='skirt', vmin=-1, vmax=2,
                              cbar_fmt='%.1f')
    fig2.tight_layout()
    rep.add_figs_to_section(fig1, captions='Trial %d (time-freq)' % idx,
                            section='time-freq')
    rep.add_figs_to_section(fig2, captions='Trial %d (topomap 40-70 Hz)' % idx,
                            section='topomap')
    if idx == 0:
        power_sum = power
    else:
        power_sum += power

power_sum.data /= n_epochs
fig1 = power_sum.plot(picks, baseline=baseline, mode=mode, vmin=-0.45,
                      vmax=0.6, layout=layout, tmin=-0.8, tmax=tmax)
fig2 = power_sum.plot_topomap(layout=layout, fmin=40., fmax=70.,
                              tmin=0.3, tmax=tmax,
                              baseline=baseline, mode=mode, show_names=False,
                              outlines='skirt', cbar_fmt='%.1f')
fig2.tight_layout()
rep.add_figs_to_section(fig1, captions='Average (time-freq)',
                        section='time-freq')
rep.add_figs_to_section(fig2, captions='Average (topomap 40-70 Hz)',
                        section='topomap')

rep.save('tf_gamm.html', overwrite=True)
