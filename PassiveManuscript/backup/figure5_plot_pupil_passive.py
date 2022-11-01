#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:25:48 2022
By: Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from serotonin_functions import paths, figure_style, load_passive_opto_times, load_subjects

# Load in data
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure5')
pupil_df = pd.read_csv(join(save_path, 'pupil_passive.csv'))

# Get median over sessions
avg_pupil_df = pupil_df.groupby(['time', 'subject']).median().reset_index()
avg_pupil_df = avg_pupil_df[avg_pupil_df['subject'] != 'ZFM-03324']  # drop this animal for now

# Plot
colors, dpi = figure_style()
f, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax.add_patch(Rectangle((0, -5), 1, 12, color='royalblue', alpha=0.25, lw=0))
g = sns.lineplot(x='time', y='baseline_subtracted', data=avg_pupil_df, ax=ax, hue='expression',
                 hue_order=[1, 0], palette=[colors['sert'], colors['wt']], errorbar='se')
ax.set(ylabel='Pupil size change (%)', xlabel='Time (s)', xticks=[0, 1, 2, 3, 4], ylim=[-4, 6])
ax.legend(frameon=False, bbox_to_anchor=(0.7, 0.8), prop={'size': 5.5})
new_labels = ['SERT', 'WT']
for t, l in zip(g.legend_.texts, new_labels):
    t.set_text(l)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'pupil_size_opto.pdf'))




