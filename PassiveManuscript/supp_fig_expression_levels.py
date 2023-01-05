#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:50:51 2022
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join, isfile
from matplotlib.patches import Rectangle
from serotonin_functions import paths, query_ephys_sessions, load_subjects, figure_style
from atlaselectrophysiology.load_histology import download_histology_data
from pathlib import Path
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()

# Settings
RAW_DATA_PATH = 'E:\\Flatiron\\mainenlab\\Subjects'
#RAW_DATA_PATH = 'D:\\Flatiron\\mainenlab\\Subjects'
AP_EXT = [-4400, -4600]
EXP_WIN_XY = [190, 130]  # top left point
CTRL_WIN_XY = [190, 85]
WIN_WIDTH = 80
WIN_HEIGHT = 30
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'supp_figure_expression')

# Query subjects
rec = query_ephys_sessions(anesthesia='all')
subjects = rec['subject'].unique()
subject_info = load_subjects()
subject_info = subject_info[subject_info['subject'].isin(subjects)]
subject_info = subject_info.sort_values(by='sert-cre', ascending=False)

colors, dpi = figure_style()
f, axs = plt.subplots(3, 6, figsize=(7, 4), dpi=dpi)
axs = np.concatenate(axs)
expr_df = pd.DataFrame()
for i, subject in enumerate(subject_info['subject']):
    print(f'Processing {subject} ({i+1} of {len(subjects)})')
    sert_cre = subject_info.loc[subject_info['subject'] == subject, 'sert-cre'].values[0]

    # Get paths to green and red channel of the histology data
    gr_path = Path(join(RAW_DATA_PATH, subject, 'histology', f'STD_ds_{subject}_GR.nrrd'))

    # Download histology if not already on disk
    if ~isfile(gr_path):
        _ = download_histology_data(subject, 'mainenlab')

    # Initialize Allen atlas objects
    gr_hist = AllenAtlas(hist_path=gr_path)

    all_rel_fluo = np.empty(np.arange(AP_EXT[0], AP_EXT[1]-25, -25).shape[0])
    for j, ap in enumerate(np.arange(AP_EXT[0], AP_EXT[1]-25, -25)):

        slice_im = np.moveaxis(gr_hist.slice(ap/1e6, axis=1), 0, 1)

        test_slice = slice_im[EXP_WIN_XY[1]:EXP_WIN_XY[1]+WIN_HEIGHT, EXP_WIN_XY[0]:WIN_WIDTH+EXP_WIN_XY[0]]
        control_slice = slice_im[CTRL_WIN_XY[1]:CTRL_WIN_XY[1]+WIN_HEIGHT, CTRL_WIN_XY[0]:WIN_WIDTH+EXP_WIN_XY[0]]

        all_rel_fluo[j] = (np.sum(test_slice[test_slice > np.percentile(test_slice, 99)])
                           / np.sum(control_slice[control_slice > np.percentile(control_slice, 99)]))

    rel_fluo = (np.max(all_rel_fluo) * 100) - 100

    # Plot figures
    slice_im = np.moveaxis(gr_hist.slice(
        np.arange(AP_EXT[0], AP_EXT[1]-25, -25)[np.argmax(all_rel_fluo)]/1e6, axis=1), 0, 1)
    axs[i].imshow(slice_im, cmap='bone', vmin=0,
                  vmax=np.mean(slice_im)+(np.std(slice_im)*4))
    axs[i].add_patch(Rectangle((EXP_WIN_XY[0], EXP_WIN_XY[1]), WIN_WIDTH, WIN_HEIGHT, fill=False,
                               edgecolor='white', lw=0.5, ls='--'))
    axs[i].add_patch(Rectangle((CTRL_WIN_XY[0], CTRL_WIN_XY[1]), WIN_WIDTH, WIN_HEIGHT, fill=False,
                               edgecolor='white', lw=0.5, ls='--'))
    axs[i].axis('off')
    axs[i].set(ylim=[200, 40], xlim=[150, 300])
    if sert_cre == 1:
        axs[i].set(title=f'{subject}, SERT')
    else:
        axs[i].set(title=f'{subject}, WT')

    # Add to dataframe
    expr_df = pd.concat((expr_df, pd.DataFrame(index=[expr_df.shape[0]+1], data={
        'subject': subject, 'sert-cre': sert_cre, 'rel_fluo': rel_fluo})))

for i in range(i+1, len(axs)-1):
    axs[i].axis('off')

# Save output
expr_df.to_csv(join(save_path, 'expression_levels.csv'))

# Plot overview plot

f.subplots_adjust(bottom=0.3, left=0.32, right=0.88, top=0.9)
sns.swarmplot(x='sert-cre', y='rel_fluo', data=expr_df, order=[1, 0], size=3,
              palette=[colors['sert'], colors['wt']], ax=axs[-1])
axs[-1].set(xticklabels=['SERT', 'WT'], ylabel='Relative fluoresence (%)', xlabel='',
            yticks=[0, 100, 200, 300, 400])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'supp_fig_expression_levels.pdf'))
