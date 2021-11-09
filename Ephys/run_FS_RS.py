#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

from os.path import join
from serotonin_functions import paths, figure_style
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

# Settings
_, _, data_dir = paths()
FEATURES = ['spike_amp', 'pt_ratio', 'rp_slope', 'rc_slope', 'peak_to_trough', 'spike_width', 'firing_rate']
#FEATURES = ['spike_width', 'firing_rate']

waveforms_df = pd.read_pickle(join(data_dir, 'waveform_metrics.p'))

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42).fit(waveforms_df[FEATURES].to_numpy())
neuron_types = kmeans.labels_

# Do TSNE embedding
tsne_emb = TSNE(n_components=2, perplexity=35, random_state=55).fit_transform(waveforms_df[FEATURES].to_numpy())

colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.5), dpi=dpi)
ax1.scatter(waveforms_df.loc[neuron_types == 0, 'firing_rate'], waveforms_df.loc[neuron_types == 0, 'spike_width'], label='RS')
ax1.scatter(waveforms_df.loc[neuron_types == 1, 'firing_rate'], waveforms_df.loc[neuron_types == 1, 'spike_width'], label='FS')
ax1.set(xlabel='Firing rate (spks/s)', ylabel='Spike width (ms)')

ax2.scatter(tsne_emb[neuron_types == 0, 0], tsne_emb[neuron_types == 0, 1])
ax2.scatter(tsne_emb[neuron_types == 1, 0], tsne_emb[neuron_types == 1, 1])

