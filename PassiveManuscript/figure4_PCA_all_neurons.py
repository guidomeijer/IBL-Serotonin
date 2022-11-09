# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:25:01 2022

@author: Guido Meijer
"""

import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from serotonin_functions import paths, load_subjects, figure_style
from dlc_functions import smooth_interpolate_signal_sg

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure4')

# Load in data
pca_df = pd.read_csv(join(save_path, 'pca_all_neurons.csv'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    pca_df.loc[pca_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# for

# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, 0), 1, 100, color='royalblue', alpha=0.25, lw=0))
#ax.add_patch(Rectangle((0, 0), 1, -100, color='royalblue', alpha=0.25, lw=0))
#sns.lineplot(x='time', y='pca_dist', ax=ax1, legend=None, hue='sert-cre', ci=68,
#             data=pca_df[pca_df['sert-cre'] == 1], palette=['k'])
sns.lineplot(x='time', y='pca_dist', ax=ax1, legend='brief', hue='sert-cre', errorbar='se',
             data=pca_df, hue_order=[0, 1], palette=[colors['wt'], colors['sert']])
ax1.set(xlim=[-1, 4], xlabel='Time (s)', ylabel='PCA trajectory distance\nfrom baseline (a.u.)',
        xticks=[-1, 0, 1, 2, 3, 4], ylim=[5, 15], yticks=[5, 10, 15])
leg_handles, _ = ax1.get_legend_handles_labels()
leg_labels = ['WT', 'SERT']
leg = ax1.legend(leg_handles, leg_labels, prop={'size': 5}, loc='upper right', title='')
leg.get_frame().set_linewidth(0.0)

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'PCA_opto_all_neurons.pdf'))

# %%

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, 0), 1, 100, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='pca_dist_spont', ax=ax1, legend='brief', errorbar='se',
             data=pca_df[pca_df['sert-cre'] == 1], color='grey', label='Control',
             err_kws={'linewidth': 0, 'alpha': 0.3})
sns.lineplot(x='time', y='pca_dist', ax=ax1, legend='brief', errorbar='se',
             data=pca_df[pca_df['sert-cre'] == 1], color=colors['sert'], label='5-HT stim',
             err_kws={'linewidth': 0, 'alpha': 0.3})
ax1.set(xlim=[-1, 4], xlabel='Time (s)', ylabel='PCA trajectory distance\nfrom baseline (a.u.)',
        xticks=[-1, 0, 1, 2, 3, 4], ylim=[4, 14], yticks=[4, 14])
leg = ax1.legend(prop={'size': 5}, loc='upper right', title='')
leg.get_frame().set_linewidth(0.0)

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'PCA_opto_spont_all_neurons.pdf'))







