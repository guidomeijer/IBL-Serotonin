#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:08:03 2020

@author: guido
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ibl_pipeline import subject, acquisition, behavior

# Get stim prob left from example session
b = ((subject.Subject & 'subject_nickname="KS014"') * behavior.TrialSet.Trial
     * (acquisition.Session & 'task_protocol LIKE "%biased%"'
        & 'session_start_time BETWEEN "2019-08-30" and "2019-08-31"'))
prob_left = b.fetch('trial_stim_prob_left')


    
# %% Plot

# Get example stimulated trials
stim_on = np.ones(90) * np.nan
while stim_on.shape[0] < prob_left.shape[0]:
    block_length = int(np.random.exponential(60))
    while (block_length < 20) | (block_length > 100):
        block_length = int(np.random.exponential(60))
    if (stim_on.shape[0] == 90) & (np.random.choice([0, 1]) == 1):
        stim_on = np.append(stim_on, np.ones(block_length) * 0.85)
    elif (stim_on.shape[0] > 90) & np.isnan(stim_on[-1]):
        stim_on = np.append(stim_on, np.ones(block_length) * 0.85)
    elif (stim_on.shape[0] > 90) & ~np.isnan(stim_on[-1]):
        stim_on = np.append(stim_on, np.ones(block_length) * np.nan)
    else:
        stim_on = np.append(stim_on, np.ones(block_length) * np.nan)
stim_on = stim_on[:prob_left.shape[0]]

sns.set(style="ticks", context="talk", font_scale=2)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 12))
ax1.plot(np.arange(1, prob_left.shape[0] + 1), prob_left, color=[0.6, 0.6, 0.6], lw=3)
ax1.plot(np.arange(1, prob_left.shape[0] + 1), stim_on, 'b', lw=5)
ax1.set(xlabel='Trials', ylabel='Probability of left stimulus')

sns.despine(trim=True)