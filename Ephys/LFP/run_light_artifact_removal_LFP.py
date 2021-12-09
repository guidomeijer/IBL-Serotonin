#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, isfile
import brainbox.io.one as bbone
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 load_lfp, remap)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
PLOT = True
OVERWRITE = True
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'LFP')
save_path = join(save_path, 'LFP')

# Query sessions
eids, _, subjects = query_ephys_sessions(return_subjects=True, one=one)

for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]

    print(f'Starting {subject}, {date}')

    # Load in laser pulse times
    try:
        opto_train_times, opto_on_times, opto_off_times = load_passive_opto_times(
                                                            eid, return_off_times=True, one=one)
    except:
        print('Session does not have passive laser pulses')
        continue
    if len(opto_train_times) == 0:
        print('Did not find ANY laser pulses!')
        continue
    else:
        print(f'Found {len(opto_train_times)} passive laser pulses')

    # Load in channels
    channels = bbone.load_channel_locations(eid, one=one)

    for p, probe in enumerate(channels.keys()):

        # Check if artifact removal has already been done
        if (isfile(join(save_path, f'{subject}_{date}_{probe}_cleaned_lfp.npy'))) & ~OVERWRITE:
            continue

        # Load in lfp
        lfp, time = load_lfp(eid, probe, time_start=opto_on_times[0]-10, time_end=opto_on_times[-1]+10,
                             relative_to='begin', one=one)

        # Remove light artifacts
        print('Remove light artifacts from LFP trace')
        for of, p_time in enumerate(np.sort(np.concatenate((opto_on_times, opto_off_times)))):
            pulse_ind = np.argmin(np.abs(time - p_time))
            lfp = np.delete(lfp, np.arange(pulse_ind-5, pulse_ind+5), axis=1)
            time = np.delete(time, np.arange(pulse_ind-5, pulse_ind+5))






        for ch in range(lfp.shape[0]):
            if np.mod(ch, 24) == 0:
                print(f'Channel {ch+1} of {lfp.shape[0]}..')
            for of, p_time in enumerate(np.sort(np.concatenate((opto_on_times, opto_off_times)))):
                pulse_ind = np.argmin(np.abs(time - p_time))
                asd

                lfp_copy[ch, pulse_ind+5:] = lfp[ch, pulse_ind+5:] + (lfp[ch, pulse_ind-5] - lfp[ch, pulse_ind+5])
                lfp_copy[ch, :] = np.delete(lfp_copy[ch, :], np.arange(pulse_ind-5, pulse_ind+5))
                time = np.delete(time, np.arange(pulse_ind-5, pulse_ind+5))

        np.save(join(save_path, f'{subject}_{date}_{probe}_cleaned_lfp'), lfp)
        np.save(join(save_path, f'{subject}_{date}_{probe}_timestamps'), time)


