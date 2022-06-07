# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 10:01:53 2022

@author: Guido
"""

from serotonin_functions import paths, figure_style, plot_scalar_on_slice
import matplotlib.pyplot as plt
from ibllib.atlas import AllenAtlas
ba = AllenAtlas(res_um=10)


# Plot brain map slices
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 4), dpi=300)

plot_scalar_on_slice(['ORBl', 'ORBm'], [], ax=ax1,
                     slice='coronal', coord=2*1000, brain_atlas=ba)
ax1.axis('off')


plot_scalar_on_slice([], [], ax=ax2,
                     slice='coronal', coord=2.2*1000, brain_atlas=ba)
ax2.axis('off')