#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, isfile
import brainbox.io.one as bbone
import matplotlib.pyplot as plt
from serotonin_functions import (paths, query_ephys_sessions, load_passive_opto_times, load_lfp,
                                 figure_style)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
PLOT = True
OVERWRITE = True
FS = 2500  # sampling freq
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'LFP', 'ArtifactRemoval')
save_path = join(save_path, 'LFP')

# Query sessions
rec = query_ephys_sessions(one=one)

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
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
        # Get average shape of light artifact
        for of, p_time in enumerate(opto_on_times):
            pulse_ind = np.argmin(np.abs(time - p_time))
            if of == 0:
                lfp_pulse = lfp[:, pulse_ind : pulse_ind + 30].copy()
            else:
                #lfp_pulse = np.dstack((lfp_pulse, lfp[:, pulse_ind : pulse_ind + 30]))
                lfp_pulse += lfp[:, pulse_ind : pulse_ind + 30]
        lfp_pulse = lfp_pulse / of

        # Remove average light artifact shape from each pulse
        lfp_cleaned = lfp.copy()
        for of, p_time in enumerate(opto_on_times):
            pulse_ind = np.argmin(np.abs(time - p_time))
            lfp_cleaned[:, pulse_ind : pulse_ind + 30] = (lfp_cleaned[:, pulse_ind : pulse_ind + 30]
                                                          - lfp_pulse)

        # Plot cleaning output
        colors, dpi = figure_style()
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4), dpi=dpi)
        ax1.plot(lfp_pulse[10,:])
        ax1.set(title='Mean artifact')
        ax1.axis('off')
        ax2.plot(time[FS*10:int(FS*10.5)], lfp[10, FS*10:int(FS*10.5)])
        ax2.set(title='Original')
        ax2.axis('off')
        ax3.plot(time[FS*9:FS*11], lfp_cleaned[10, FS*9:FS*11])
        ax3.set(title='Artifacts removed')
        plt.axis('off')
        plt.savefig(join(fig_path, f'{subject}_{date}_{probe}.jpg'), dpi=300)
        plt.close(f)

        # Save results
        np.save(join(save_path, f'{subject}_{date}_{probe}_cleaned_lfp'), lfp_cleaned)
        np.save(join(save_path, f'{subject}_{date}_{probe}_timestamps'), time)


