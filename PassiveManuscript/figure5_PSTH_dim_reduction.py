# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:30:06 2022

@author: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from serotonin_functions import paths, figure_style, load_subjects
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
svd = TruncatedSVD(n_components=2, random_state=42)
pca = PCA(n_components=2, random_state=42)
tsne = TSNE(n_components=2, random_state=42)

# Paths
fig_path, data_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive')

# Load in PSTH data
psth_df = pd.read_pickle(join(data_path, 'psth.pickle'))

# Select SERT mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    psth_df.loc[psth_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
psth_df = psth_df[psth_df['sert-cre'] == 1]

# Transform into 2D array (time bins x neurons)
all_psth = np.column_stack(psth_df['peth'].to_numpy()).T

"""
# Divide over baseline
for i in range(all_psth.shape[0]):
    all_psth[i, :] = all_psth[i, :] - np.mean(all_psth[i, psth_df.loc[0, 'time'] < 0])

"""
# Normalize PSTHs to max firing
for i in range(all_psth.shape[0]):
    all_psth[i, :] = all_psth[i, :] / np.max(all_psth[i, :])


# Perform dimensionality reduction
pca.fit(all_psth)
dim_red_psth = pca.transform(all_psth)

# Add to dataframe
psth_df['dim_1'] = dim_red_psth[:, 0]
psth_df['dim_2'] = dim_red_psth[:, 1]

plt.scatter(dim_red_psth[:, 0], dim_red_psth[:, 1])

"""
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(dim_red_psth[:, 0], dim_red_psth[:, 1], dim_red_psth[:, 2])
"""