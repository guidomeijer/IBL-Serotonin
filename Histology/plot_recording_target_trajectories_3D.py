#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:40:01 2020

@author: guido
"""

import pathlib
import pandas as pd
import seaborn as sns
from mayavi import mlab
from os.path import join
import ibllib.atlas as atlas
from atlaselectrophysiology import rendering
ba = atlas.AllenAtlas(25)

# Load in recording target coordinates
rec_targets = pd.read_csv(join(pathlib.Path(__file__).parent.absolute(), '..', 'recording_targets.csv'))

# Convert to meters
rec_targets[['ML', 'AP', 'DV', 'depth']] = rec_targets[['ML', 'AP', 'DV', 'depth']].divide(1000000)

# Render 3D plot with trajectories
fig = rendering.figure(grid=False)
colors = sns.color_palette('colorblind', rec_targets['Craniotomy'].unique().shape[0])
for i in rec_targets.index:
    ins = atlas.Insertion(x=rec_targets.loc[i, 'ML'], y=rec_targets.loc[i,'AP'],
                          z=rec_targets.loc[i,'DV'], phi=rec_targets.loc[i,'phi'],
                          theta=rec_targets.loc[i,'theta'], depth=rec_targets.loc[i,'depth'])
    mlapdv = ba.xyz2ccf(ins.xyz)
    mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                line_width=1, tube_radius=25, color=colors[rec_targets.loc[i, 'Craniotomy'] - 1])

# Add fiber to plot
fiber = atlas.Insertion(x=0, y=-0.00664, z=-0.0005, phi=270, theta=32, depth=0.004)
mlapdv = ba.xyz2ccf(fiber.xyz)
mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                line_width=1, tube_radius=200, color=(.6, .6, .6))
