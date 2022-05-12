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
from scipy.stats import ttest_ind
from matplotlib.patches import Rectangle
from serotonin_functions import paths, load_subjects, figure_style
from dlc_functions import smooth_interpolate_signal_sg

# Settings
REGION_PAIRS = ['M2-ORB', 'M2-mPFC']

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure5')

# Load in data
cca_df = pd.read_csv(join(save_path, 'cca_results_all.csv'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    cca_df.loc[cca_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Take slice of dataframe
cca_df = cca_df[(cca_df['sert-cre'] == 1) & cca_df['region_pair'].isin(REGION_PAIRS)]

# Smooth traces (this is ugly I know)
for i, eid in enumerate(cca_df['eid'].unique()):
    for r, region in enumerate(REGION_PAIRS):
        if cca_df.loc[(cca_df['eid'] == eid) & (cca_df['region_pair'] == region)].shape[0] > 0:
            cca_df.loc[(cca_df['eid'] == eid) & (cca_df['region_pair'] == region), 'r_baseline_smooth'] = (
                smooth_interpolate_signal_sg(cca_df.loc[(cca_df['eid'] == eid)
                                                        & (cca_df['region_pair'] == region), 'r_baseline'], window=5, order=1))
            cca_df.loc[(cca_df['eid'] == eid) & (cca_df['region_pair'] == region), 'r_smooth'] = (
                smooth_interpolate_signal_sg(cca_df.loc[(cca_df['eid'] == eid)
                                                        & (cca_df['region_pair'] == region), 'r_opto'], window=5, order=1))

# Do statistics
cca_table_df = cca_df.pivot(index='time', columns=['region_pair', 'eid'], values='r_baseline_smooth')
cca_table_df = cca_table_df.reset_index()
p_values = ttest_ind(cca_table_df['M2-mPFC'].to_numpy(), cca_table_df['M2-ORB'].to_numpy(), axis=1)[1]

# %%
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, 0), 1, 0.6, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='r_smooth', data=cca_df, hue_order=REGION_PAIRS,
             ax=ax1, hue='region_pair', palette=[colors[i] for i in REGION_PAIRS], ci=68)
#ax1.plot(ax1.get_xlim(), [0, 0], ls='--', color='grey')
ax1.plot(cca_table_df['time'][p_values < 0.05], [0.2, 0.2], color='k')
ax1.set(ylabel='Population correlation (r)', ylim=[0, 0.6], yticks=[0, .1, .2, .3, .4, .5, .6],
        xticks=[-1, 0, 1, 2, 3], xlabel='Time (s)')
leg = ax1.legend(loc='lower left', prop={'size': 6})
leg.get_frame().set_linewidth(0.0)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'CCA_M2_mPFC_ORB.pdf'))
