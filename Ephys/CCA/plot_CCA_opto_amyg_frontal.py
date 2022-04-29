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

fig_path, save_path = paths()

# Load in data
cca_df = pd.read_csv(join(save_path, 'cca_results_front_amyg.csv'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    cca_df.loc[cca_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

cca_df['over_spont'] = cca_df['r_opto'] - cca_df['r_spont_mean']

# Plot
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
sns.lineplot(x='time', y='over_spont', hue='region_pair', data=cca_df[cca_df['sert-cre'] == 1], ax=ax1,
             palette='tab10')
ax1.set(ylabel='Population correlation (r)', xlabel='Time (s)')
ax1.legend(frameon=False)

sns.lineplot(x='time', y='over_spont', hue='region_pair', data=cca_df[cca_df['sert-cre'] == 0], ax=ax2,
             palette='tab10')
ax2.set(ylabel='Population correlation (r)', xlabel='Time (s)')
ax2.legend(frameon=False)

plt.tight_layout()


