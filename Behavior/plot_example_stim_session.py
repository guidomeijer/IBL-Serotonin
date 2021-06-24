#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:08:03 2020

@author: guido
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from brainbox.task.closed_loop import generate_pseudo_blocks
from serotonin_functions import figure_style

N_TRIALS = 400

prob_left = generate_pseudo_blocks(N_TRIALS, first5050=0)
stim_on = generate_pseudo_blocks(N_TRIALS, first5050=0)
stim_on[stim_on == 0.2] = np.nan
stim_on[stim_on == 0.8] = 0.85

figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(6, 5), dpi=150)
ax1.plot(np.arange(1, prob_left.shape[0] + 1), prob_left, color=[0.6, 0.6, 0.6], lw=2)
ax1.plot(np.arange(1, prob_left.shape[0] + 1), stim_on, 'b', lw=3)
ax1.set(xlabel='Trials', ylabel='Probability of left stimulus')

sns.despine(trim=True)
plt.tight_layout()