#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:35:50 2022
By: Guido Meijer
"""

import numpy as np
import seaborn as sns
import ssm
from ssm.plots import gradient_cmap
import matplotlib.pyplot as plt
from serotonin_functions import figure_style
from dlc_functions import smooth_interpolate_signal_sg


T = 100  # number of time bins
K = 3    # number of discrete states
D = 50   # dimension of the observations
N_FRAMES = 5000  # nr of frames to use
FRAME_RATE = 60 # sampling frequency

# Load in facemap data
fm_dict = np.load('/media/guido/Data2/Facemap/ZFM-01802_2021-03-09_left_shorts_proc.npy',
                  allow_pickle=True).item()

# Make an hmm and sample from it
arhmm = ssm.HMM(K, D, observations="ar")
arhmm.fit(fm_dict['motSVD'][1][:N_FRAMES, :D])
zhat = arhmm.most_likely_states(fm_dict['motSVD'][1][:N_FRAMES, :D])

# Get transition matrix
transition_mat = arhmm.transitions.transition_matrix

# %% Plot
color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green"
    ]
colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)
colors, dpi = figure_style()

time_ax = np.linspace(0, N_FRAMES/FRAME_RATE, N_FRAMES)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.5, 1.75),
                                    gridspec_kw={'width_ratios':[1, 1, 0.7]}, dpi=dpi)
ax1.imshow(zhat[None,:], aspect="auto",
           extent=[0, N_FRAMES/FRAME_RATE, fm_dict['motSVD'][1][:N_FRAMES, 0].min(), fm_dict['motSVD'][1][:N_FRAMES, 0].max()],
           cmap=cmap, vmin=0, vmax=K)
ax1.plot(time_ax, fm_dict['motSVD'][1][:N_FRAMES, 0], zorder=1, color='k', lw=0.5)
ax1.set(xlabel='Time (s)', ylabel='First SVD dimension')

ax2.imshow(zhat[None,:], aspect="auto",
           extent=[0, N_FRAMES/FRAME_RATE, fm_dict['motSVD'][1][:N_FRAMES, 0].min(), fm_dict['motSVD'][1][:N_FRAMES, 0].max()],
           cmap=cmap, vmin=0, vmax=K)
ax2.plot(time_ax, fm_dict['motSVD'][1][:N_FRAMES, 0], zorder=1, color='k', lw=0.5)
ax2.set(xlabel='Time (s)', ylabel='First SVD dimension', xlim=[0, 5])

im = ax3.imshow(transition_mat, cmap='gray')
ax3.set(title="Transition Matrix")

cbar_ax = fig.add_axes([0.955, 0.25, 0.01, 0.6])
fig.colorbar(im, cax=cbar_ax)

plt.tight_layout()
