#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:05:42 2022
By: Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from serotonin_functions import paths, load_subjects, figure_style

# Get paths
fig_path, save_path = paths()

# Load in data
state_df = pd.read_csv(join(save_path, 'state_change_face.csv'))
subjects = load_subjects()

# Add which subjects had a fixed ISI
for i, subject in enumerate(np.unique(state_df['subject'])):
    state_df.loc[state_df['subject'] == subject, 'fixed_isi'] = subjects.loc[
        subjects['subject'] == subject, 'fixed_isi'].values[0]

# Plot results
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)

sns.lineplot(x='time', y='change_rate', data=state_df[state_df['fixed_isi'] == 1], hue='sert-cre', errorbar='se',
             hue_order=[0, 1], palette=[colors['wt'], colors['sert']], ax=ax1)
ylim = ax1.get_ylim()
ax1.add_patch(Rectangle((0, -100), 1, 200, color='royalblue', alpha=0.25, lw=0))
ax1.set(xlabel='Time (s)', ylabel='State changes per second', ylim=ylim, xticks=[-1, 0, 1, 2, 3],
        yticks=[0, 2, 4, 6, 8], xlim=[-1, 3], title='Fixed inter-stimulus interval')
leg_handles, _ = ax1.get_legend_handles_labels()
leg_labels = ['WT', 'SERT']
leg = ax1.legend(leg_handles, leg_labels, prop={'size': 5}, loc='lower right', title='')
leg.get_frame().set_linewidth(0.0)

sns.lineplot(x='time', y='change_rate', data=state_df[state_df['fixed_isi'] == 0], hue='sert-cre', errorbar='se',
             hue_order=[0, 1], palette=[colors['wt'], colors['sert']], ax=ax2)
ylim = ax2.get_ylim()
ax2.add_patch(Rectangle((0, -100), 1, 200, color='royalblue', alpha=0.25, lw=0))
ax2.set(xlabel='Time (s)', ylabel='State changes per second', ylim=ylim, xticks=[-1, 0, 1, 2, 3],
        yticks=[0, 2, 4, 6, 8], xlim=[-1, 3], title='Random inter-stimulus interval')
leg_handles, _ = ax2.get_legend_handles_labels()
leg_labels = ['WT', 'SERT']
leg = ax2.legend(leg_handles, leg_labels, prop={'size': 5}, loc='upper right', title='')
leg.get_frame().set_linewidth(0.0)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'state_change_face.jpg'), dpi=600)