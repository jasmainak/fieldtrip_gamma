"""
Reproduce FieldTrip example here:
http://www.fieldtriptoolbox.org/tutorial/beamformingextended/
"""

# Authors: Mainak Jas <mainakjas@gmail.com>

import numpy as np

from mne import create_info, EpochsArray
from mne.io import read_epochs_fieldtrip

from mne.externals.pymatreader.pymatreader import read_mat

ch_names = ['MLF34', 'EMGrgt', 'MRP31', 'MRF22', 'MLC13', 'MRC12',
            'MRT25', 'MRC14', 'MLT14', 'MLO22', 'MLT25', 'MLC41',
            'MRC32', 'MLC21', 'MLO43', 'MRT31', 'MRF32', 'MLO33',
            'MRF51', 'MRP21', 'MLC11', 'MLP21', 'MLT35', 'MRC23',
            'MZP02', 'MRT22', 'MZF02', 'MZO01', 'MRP32', 'MLC15',
            'MLT21', 'MZC01', 'MZP01', 'MLC33', 'MLC42', 'MRO32',
            'MLF32', 'MRT23', 'MRT16', 'MLO41', 'MLF23', 'MLF31',
            'MRO22', 'MRF43', 'MLT41', 'MRT32', 'MLP12', 'MRT26',
            'MRT12', 'MRT44', 'MLF22', 'MLF52', 'MRT42', 'MLT13',
            'MRT21', 'MLP31', 'MLP22', 'MLT11', 'MRF42', 'MLT23',
            'MLO12', 'MRT13', 'MLT24', 'MLF42', 'MLT16', 'MLF41',
            'MRP12', 'EMGlft', 'MRO11', 'MRF12', 'MRT11', 'MRC43',
            'MLF44', 'MLC43', 'MRP11', 'MZC02', 'MLP13', 'MRT43',
            'MRC24', 'MLT12', 'MLF43', 'MLT22', 'MLF45', 'MZF03',
            'MLC24', 'MLP11', 'MLT43', 'MRO31', 'MLT15', 'MLP32',
            'MRF34', 'MLC14', 'MRF41', 'MLT34', 'MRT41', 'MLF51',
            'MRT24', 'MRT34', 'MLT44', 'MRF11', 'MZF01', 'MRF33',
            'MRT33', 'MLF12', 'MLP34', 'MRC41', 'MLT26', 'MRP34',
            'MLC12', 'MLO31', 'MLT42', 'MRC21', 'MRP13', 'MZO02',
            'MRP22', 'MLO11', 'MRC22', 'MRO41', 'MRP33', 'MRC33',
            'MRC13', 'MRF23', 'MLO32', 'MRC15', 'MRF44', 'MRF31',
            'MRT15', 'MRO33', 'MLF21', 'MLP33', 'MRF52', 'MLC31',
            'MLF33', 'MLF11', 'MLO42', 'MLC32', 'MRO43', 'MLO21',
            'MRC42', 'MRC11', 'MRT14', 'MRO21', 'MLC23', 'MRO12',
            'MLT32', 'MLT33', 'MRC31', 'MRF45', 'MRO42', 'MLC22',
            'MRF21', 'MRT35', 'MLT31']
sfreq = 400.

info = create_info(ch_names, sfreq, ch_types='eeg')
# epochs = read_epochs_fieldtrip('subjectK.mat', info=info,
#                                data_name='data_left')

ft_struct = read_mat('subjectK.mat',
                   ignore_fields=['previous'],
                   variable_names=['data_left'])
data = ft_struct['data_left']['trial']

n_trials = len(data)
min_n_times = min([d.shape[1] for d in data])
n_channels = len(ch_names)

data_epochs = np.empty((n_trials, n_channels, min_n_times))

for idx, d in enumerate(data):
      data_epochs[idx, :, :min_n_times] = data[idx][:, :min_n_times]

epochs = EpochsArray(data_epochs, info)
epochs.plot(scalings=dict(eeg=1.5e-12))
