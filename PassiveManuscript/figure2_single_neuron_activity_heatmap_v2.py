#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from serotonin_functions import figure_style
from brainbox.io.one import SpikeSortingLoader
from brainbox.singlecell import calculate_peths
from serotonin_functions import (paths, load_passive_opto_times, combine_regions, load_subjects,
                                 high_level_regions)

# Settings
T_BEFORE = 1  # for plotting
T_AFTER = 2
VMIN = 0
VMAX = 2
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure2')

# Load in data
peths_df = pd.read_pickle(join(save_path, 'psth.pickle'))
peths_df = peths_df.reset_index(drop=True)  # make sure index is reset
peths_df = peths_df.sort_values(['region', 'modulation'], ascending=[True, False])  # sort by modulation

# Get array of all PETHs and select time limits
time_ax = peths_df['time'][0]
all_peth = np.array(peths_df['peth'].tolist())
all_peth = all_peth[:, (time_ax > -T_BEFORE) & (time_ax < T_AFTER)]
time_ax = time_ax[(time_ax > -T_BEFORE) & (time_ax < T_AFTER)]

# Do baseline normalization
norm_peth = np.empty(all_peth.shape)
for i in range(all_peth.shape[0]):
    norm_peth[i, :] = all_peth[i, :] / (np.mean(all_peth[i, time_ax < 0]) + 0.1)


# %%
# Plot per region
colors, dpi = figure_style()
f, ((ax_pag, ax_mpfc, ax_orb, ax_str, ax_mrn),
    (ax_sc, ax_rsp, ax_am, ax_m2, ax_vis),
    (ax_th, ax_pir, ax_hc, ax_olf, ax_cb)) = plt.subplots(3, 5, figsize=(7, 3.5), sharex=True, dpi=dpi)
title_font = 7
cmap = sns.diverging_palette(240, 20, as_cmap=True)

these_peths = norm_peth[peths_df['region'] == 'Medial prefrontal cortex']
img = ax_mpfc.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_mpfc.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_mpfc.set_title('Medial prefrontal cortex', fontsize=title_font)
ax_mpfc.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = norm_peth[peths_df['region'] == 'Orbitofrontal cortex']
img = ax_orb.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_orb.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_orb.set_title('Orbitofrontal cortex', fontsize=title_font)
ax_orb.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = norm_peth[peths_df['region'] == 'Amygdala']
img = ax_am.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_am.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_am.set_title('Amygdala', fontsize=title_font)
ax_am.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = norm_peth[peths_df['region'] == 'Visual cortex']
img = ax_vis.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_vis.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_vis.set_title('Visual cortex', fontsize=title_font)
ax_vis.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = norm_peth[peths_df['region'] == 'Hippocampus']
img = ax_hc.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_hc.set(yticks=[1], yticklabels=[these_peths.shape[0]], xlabel='Time (s)')
ax_hc.set_title('Hippocampus', fontsize=title_font)
ax_hc.xaxis.set_tick_params(which='both', labelbottom=True)
ax_hc.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = norm_peth[peths_df['region'] == 'Olfactory areas']
img = ax_olf.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_olf.set(yticks=[1], yticklabels=[these_peths.shape[0]], xlabel='Time (s)')
ax_olf.set_title('Olfactory areas', fontsize=title_font)
ax_olf.xaxis.set_tick_params(which='both', labelbottom=True)
ax_olf.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = norm_peth[peths_df['region'] == 'Piriform']
img = ax_pir.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_pir.set(yticks=[1], yticklabels=[these_peths.shape[0]], xlabel='Time (s)')
ax_pir.set_title('Piriform', fontsize=title_font)
ax_pir.plot([0, 0], [-1, 1], ls='--', color='k')
ax_pir.xaxis.set_tick_params(which='both', labelbottom=True)

these_peths = norm_peth[peths_df['region'] == 'Thalamus']
img = ax_th.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_th.set(yticks=[1], yticklabels=[these_peths.shape[0]], ylabel='Mod. neurons', xlabel= 'Time (s)')
ax_th.set_title('Thalamus', fontsize=title_font)
ax_th.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = norm_peth[peths_df['region'] == 'Secondary motor cortex']
img = ax_m2.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_m2.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_m2.set_title('Secondary motor cortex', fontsize=title_font)
ax_m2.plot([0, 0], [-1, 1], ls='--', color='k')

"""
these_peths = peths_df[peths_df['region'] == 'Barrel cortex']
img = ax_bc.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_bc.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_bc.set_title('Barrel cortex', fontsize=title_font)
ax_bc.plot([0, 0], [-1, 1], ls='--', color='k')
"""

these_peths = norm_peth[peths_df['region'] == 'Periaqueductal gray']
img = ax_pag.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_pag.set(yticks=[1], yticklabels=[these_peths.shape[0]], ylabel='Mod. neurons')
ax_pag.set_title('Periaqueductal gray', fontsize=title_font)
ax_pag.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = norm_peth[peths_df['region'] == 'Superior colliculus']
img = ax_sc.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_sc.set(yticks=[1], yticklabels=[these_peths.shape[0]], ylabel='Mod. neurons')
ax_sc.set_title('Superior colliculus', fontsize=title_font)
ax_sc.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = norm_peth[peths_df['region'] == 'Midbrain reticular nucleus']
img = ax_mrn.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_mrn.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_mrn.set_title('Midbrain reticular nucleus', fontsize=title_font)
ax_mrn.plot([0, 0], [-1, 1], ls='--', color='k')


these_peths = norm_peth[peths_df['region'] == 'Retrosplenial cortex']
img = ax_rsp.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_rsp.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_rsp.set_title('Retrosplenial cortex', fontsize=title_font)
ax_rsp.plot([0, 0], [-1, 1], ls='--', color='k')


these_peths = norm_peth[peths_df['region'] == 'Tail of the striatum']
img = ax_str.imshow(these_peths, cmap=cmap,
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_str.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_str.set_title('Tail of the striatum', fontsize=title_font)
ax_str.plot([0, 0], [-1, 1], ls='--', color='k')
"""
these_peths = peths_df[peths_df['region'] == 'Substantia nigra']
img = ax_snr.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_snr.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_snr.set_title('Substantia nigra', fontsize=title_font)
ax_snr.plot([0, 0], [-1, 1], ls='--', color='k')
"""
ax_cb.axis('off')

#plt.tight_layout()
plt.subplots_adjust(left=0.06, bottom=0.1, right=0.98, top=0.9, wspace=0.4, hspace=0)

cb_ax = f.add_axes([0.84, 0.13, 0.01, 0.2])
cbar = f.colorbar(mappable=ax_mpfc.images[0], cax=cb_ax)
cbar.ax.set_ylabel('FR / baseline', rotation=270, labelpad=10)
cbar.ax.set_yticks([0, 1, 2])


#plt.tight_layout(pad=3)
plt.savefig(join(fig_path, 'heatmap_per_region.pdf'), bbox_inches='tight')



