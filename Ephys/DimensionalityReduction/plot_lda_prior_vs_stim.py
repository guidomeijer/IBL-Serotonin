#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 15:19:11 2021
By: Guido Meijer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from serotonin_functions import paths, get_full_region_name, figure_style

# Settings
COLORMAP = 'seagreen'
_, fig_path, save_path = paths()
PRE_TIME = 0
POST_TIME = 0.3
MIN_IMPROVEMENT = 5

# Load in results
decoding_prior = pd.read_csv(join(save_path, f'lda_decoding_block_{PRE_TIME}_{POST_TIME}.csv'))
decoding_stim = pd.read_csv(join(save_path, f'lda_decoding_stim_{PRE_TIME}_{POST_TIME}.csv'))

# Calculate delta
decoding_prior['delta'] = (decoding_prior['acc_block_on'] - decoding_prior['acc_block_off']) * 100
decoding_stim['delta'] = (decoding_stim['acc_block_on'] - decoding_stim['acc_block_off']) * 100

# Get per region
delta_prior = decoding_prior.groupby('region').median()['delta'].reset_index()
delta_stim = decoding_stim.groupby('region').median()['delta'].reset_index()

diff = delta_prior['delta'].abs() - delta_stim['delta'].abs()

# Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
ax1.plot([-0.5, 1], [0, 0], ls='--', color='grey')
sns.swarmplot(data=diff, color=colors['sert'])
ax1.set(ylabel='Decoding of prior over\nstimulus side (% correct)')
ax1.get_xaxis().set_visible(False)

plt.tight_layout(pad=2)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'LDA', 'lda_prior_vs_stim.png'))
plt.savefig(join(fig_path, 'Ephys', 'LDA', 'lda_prior_vs_stim.pdf'))
