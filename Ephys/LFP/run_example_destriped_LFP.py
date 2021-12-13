#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from brainbox.lfp import power_spectrum
from serotonin_functions import figure_style
import brainbox.io.one as bbone
from serotonin_functions import paths, load_passive_opto_times, load_lfp
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
SUBJECT = 'ZFM-02600'
DATE = '2021-08-27'
PROBE = 'probe00'
CHANNEL = 120
PULSE_PLOT = -3
T_BEFORE = 0.01  # for plotting
T_AFTER = 0.2
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'LFP')
save_path = join(save_path, 'LFP')

# Get eid
eid = one.search(subject=SUBJECT, date_range=[DATE, DATE], task_protocol='ephys')[0]

# Load in laser pulse times
opto_train_times, opto_on_times, opto_off_times = load_passive_opto_times(
                                                    eid, return_off_times=True, one=one)

# Load in channels
channels = bbone.load_channel_locations(eid, one=one)

# Load in lfp
lfp, time = load_lfp(eid, PROBE, time_start=opto_on_times[0]-10, time_end=opto_on_times[-1]+10,
                     relative_to='begin', one=one)

# Load in destriped LFP
cleaned_lfp, cleaned_time = load_lfp(eid, PROBE, time_start=opto_on_times[0]-10,
                                     time_end=opto_on_times[-1]+10,
                                     relative_to='begin', destriped=True, one=one)

# Calculate power spectrum
freq, psd = power_spectrum(lfp[CHANNEL, :], fs=2500)
freq, psd_cleaned = power_spectrum(cleaned_lfp[CHANNEL, :], fs=2500)

# %% Plot
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=dpi)
ax1.plot(freq, psd, label='Original')
ax1.plot(freq, psd_cleaned, label='Artifacts removed')
ax1.legend(frameon=False)
ax1.set(xlabel='Time (s)', ylabel='Power spectral density', xlim=[0, 100])

plot_pulse_times = (opto_on_times - opto_on_times[PULSE_PLOT]) * 1000
this_lfp = lfp[CHANNEL, ((time > opto_on_times[PULSE_PLOT] - T_BEFORE) & (time < opto_on_times[PULSE_PLOT] + T_AFTER))]
this_lfp = this_lfp - np.mean(this_lfp)
ax2.plot((time[((time > opto_on_times[PULSE_PLOT] - T_BEFORE) & (time < opto_on_times[PULSE_PLOT] + T_AFTER))]\
          - opto_on_times[PULSE_PLOT]) * 1000, this_lfp,
         zorder=2, label='Original')
this_clean_lfp = cleaned_lfp[CHANNEL, ((cleaned_time > opto_on_times[PULSE_PLOT] - T_BEFORE)
                                       & (cleaned_time < opto_on_times[PULSE_PLOT] + T_AFTER))]
this_clean_lfp = this_clean_lfp - np.mean(this_clean_lfp)
ax2.plot((cleaned_time[((cleaned_time > opto_on_times[PULSE_PLOT] - T_BEFORE)
                        & (cleaned_time < opto_on_times[PULSE_PLOT] + T_AFTER))]\
          - opto_on_times[PULSE_PLOT]) * 1000, this_clean_lfp,
         zorder=2, label='Artifacts removed')
y_lim = ax2.get_ylim()
if PULSE_PLOT < 0:
    plot_pulses = plot_pulse_times[PULSE_PLOT:]
else:
    plot_pulses = plot_pulse_times[PULSE_PLOT:PULSE_PLOT+10]
for pp, pulse_time in enumerate(plot_pulses):
    ax2.plot([pulse_time, pulse_time], y_lim, ls='--', color='r',
             lw=0.5, zorder=1)
ax2.set(xlabel='Time (ms)', ylabel='uV', xlim=[-10, 200])
legend = ax2.legend(frameon=True)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Example_LFP_destriping'), dpi=300)
















