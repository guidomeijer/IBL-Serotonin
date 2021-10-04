#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:08:03 2020

@author: guido
"""

import numpy as np
import seaborn as sns
from os.path import join
import matplotlib.pyplot as plt
from brainbox.task.closed_loop import generate_pseudo_blocks
from serotonin_functions import figure_style, paths

N_TRIALS = 400
_, fig_path, _ = paths()
fig_path = join(fig_path, 'Behavior')

prob_left = generate_pseudo_blocks(N_TRIALS, first5050=0)
stim_on = generate_pseudo_blocks(N_TRIALS, first5050=0)
stim_on[stim_on == 0.2] = np.nan
stim_on[stim_on == 0.8] = 0.9

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2, 1), dpi=dpi)
ax1.plot(np.arange(1, prob_left.shape[0] + 1), prob_left, color=[0.6, 0.6, 0.6])
ax1.plot(np.arange(1, prob_left.shape[0] + 1), stim_on, color='blue', lw=1.5)
ax1.set(xlabel='Trials', ylabel='P(left stim)', yticks=[0.2, 0.8], ylim=[0.1, 0.95])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'example_stim_session.pdf'))