#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:12:54 2022
By: Guido Meijer
"""

import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import paths, load_subjects, figure_style
from dlc_functions import smooth_interpolate_signal_sg

fig_path, save_path = paths()

# Load in data
cca_df = pd.read_csv(join(save_path, 'cca_results_front_amyg.csv'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    cca_df.loc[cca_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

"""
# Smooth traces (this is ugly I know)
for i, eid in enumerate(cca_df['eid'].unique()):
    for r, region in enumerate(cca_df['region_pair'].unique()):
        if cca_df.loc[(cca_df['eid'] == eid) & (cca_df['region_pair'] == region)].shape[0] > 0:
            cca_df.loc[(cca_df['eid'] == eid) & (cca_df['region_pair'] == region), 'r_baseline_smooth'] = (
                smooth_interpolate_signal_sg(cca_df.loc[(cca_df['eid'] == eid)
                                                        & (cca_df['region_pair'] == region), 'r_baseline'], window=11))
"""

# %%
colors, dpi = figure_style()
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(7, 1.75), dpi=dpi)
sns.lineplot(x='time', y='r_baseline', data=cca_df[(cca_df['sert-cre'] == 1) & (cca_df['region_pair'] == 'M2-mPFC')], ax=ax1,
             palette='tab10', ci=68)
ax1.plot(ax1.get_xlim(), [0, 0], ls='--', color='grey')
ax1.set(title='M2-mPFC', ylabel='Population correlation (r)', ylim=[-0.15, 0.2], yticks=[-0.1, 0, 0.1, 0.2])

sns.lineplot(x='time', y='r_baseline', data=cca_df[(cca_df['sert-cre'] == 1) & (cca_df['region_pair'] == 'M2-ORB')], ax=ax2,
             palette='tab10', ci=68)
ax2.plot(ax1.get_xlim(), [0, 0], ls='--', color='grey')
ax2.set(title='M2-ORB', ylabel='Population correlation (r)', ylim=[-0.15, 0.2], yticks=[-0.1, 0, 0.1, 0.2])

sns.lineplot(x='time', y='r_baseline', data=cca_df[(cca_df['sert-cre'] == 1) & (cca_df['region_pair'] == 'mPFC-ORB')], ax=ax3,
             palette='tab10', ci=68)
ax3.plot(ax1.get_xlim(), [0, 0], ls='--', color='grey')
ax3.set(title='mPFC-ORB', ylabel='Population correlation (r)', ylim=[-0.15, 0.2], yticks=[-0.1, 0, 0.1, 0.2])

sns.lineplot(x='time', y='r_baseline', data=cca_df[(cca_df['sert-cre'] == 1) & (cca_df['region_pair'] == 'Amyg-mPFC')], ax=ax4,
             palette='tab10', ci=68)
ax4.plot(ax2.get_xlim(), [0, 0], ls='--', color='grey')
ax4.set(title='Amyg-mPFC', ylabel='Population correlation (r)', ylim=[-0.15, 0.2], yticks=[-0.1, 0, 0.1, 0.2])

plt.tight_layout()
sns.despine(trim=True)