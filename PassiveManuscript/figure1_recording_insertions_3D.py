#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:40:01 2020

@author: guido
"""

import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
from mayavi import mlab
from os.path import join
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import ibllib.atlas as atlas
from atlaselectrophysiology import rendering
from serotonin_functions import query_ephys_sessions, paths, figure_style, load_subjects
from one.api import ONE
ba = atlas.AllenAtlas(25)
one = ONE()

# Set colormap
CMAP = 'tab20'

# Paths
fig_path, _ = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure1')

# Load in recording target coordinates
rec_targets = pd.read_csv(join(pathlib.Path(__file__).parent.absolute(), '..', 'recording_targets.csv'))

# Convert to meters
rec_targets[['ML', 'AP', 'DV', 'depth']] = rec_targets[['ML', 'AP', 'DV', 'depth']].divide(1000000)

# Plot all insertions
rec = query_ephys_sessions(anesthesia='all')
fig = rendering.figure(grid=False, size=(1024, 768))
subjects = load_subjects()
colors = sns.color_palette(CMAP, subjects.shape[0])
for i, pid in enumerate(rec['pid']):
    ins_q = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                          probe_insertion=pid)
    ins = atlas.Insertion(x=ins_q[0]['x'] / 1000000, y=ins_q[0]['y'] / 1000000,
                          z=ins_q[0]['z'] / 1000000, phi=ins_q[0]['phi'],
                          theta=ins_q[0]['theta'], depth=ins_q[0]['depth'] / 1000000)
    mlapdv = ba.xyz2ccf(ins.xyz)
    mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                line_width=1, tube_radius=40,
                color=colors[subjects.loc[subjects['subject'] == rec.loc[i, 'subject']].index[0]])

# Add fiber to plot
fiber = atlas.Insertion(x=0, y=-0.00664, z=-0.0005, phi=270, theta=32, depth=0.004)
mlapdv = ba.xyz2ccf(fiber.xyz)
mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
            line_width=1, tube_radius=150, color=(.6, .6, .6))

rendering.rotating_video(join(fig_path, 'rotation_brain_insertions.avi'), fig, fps=30, secs=12)

# %%
colors, dpi = figure_style()
f, ax = plt.subplots(1, 1, figsize=(2.5, 1), dpi=dpi)
sns.heatmap([np.arange(len(subjects))], cmap=CMAP, cbar=False, ax=ax)
ax.set(yticks=[], xticks=np.arange(len(subjects))+0.5, xticklabels=np.arange(len(subjects)) + 1)
ax.set_xlabel('Mice', labelpad=2)
plt.tight_layout()
plt.savefig(join(fig_path, 'colorbar.pdf'))
