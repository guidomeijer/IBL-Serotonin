#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:50:51 2022
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join, isfile
from serotonin_functions import paths, query_ephys_sessions, load_subjects
from atlaselectrophysiology.load_histology import download_histology_data
from pathlib import Path
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()

# Settings
RAW_DATA_PATH = '/media/guido/Data2/FlatIron/mainenlab/Subjects/'
AP_EXT = [-4400, -4600]
CONTROL_REGIONS = ['SOC']

# Query subjects
rec = query_ephys_sessions()
subjects = rec['subject'].unique()
subject_info = load_subjects()

expr_df = pd.DataFrame()
for i, subject in enumerate(subjects):
    print(f'Processing {subject} ({i+1} of {len(subjects)})')

    # Get paths to green and red channel of the histology data
    gr_path = Path(join(RAW_DATA_PATH, subject, 'histology', f'STD_ds_{subject}_GR.nrrd'))

    # Download histology if not already on disk
    if ~isfile(gr_path):
        _ = download_histology_data(subject, 'mainenlab')

    # Initialize Allen atlas objects
    gr_hist = AllenAtlas(hist_path=gr_path)

    all_rel_fluo = np.empty(np.arange(AP_EXT[0], AP_EXT[1]-25, -25).shape[0])
    for j, ap in enumerate(np.arange(AP_EXT[0], AP_EXT[1]-25, -25)):

        slice_im = gr_hist.slice(ap/1e6, axis=1)
        slice_regions = gr_hist.slice(ap/1e6, axis=1, volume='value',
                                      region_values=ba.regions.id2acronym(ba.regions.remap(ba.regions.id)))

        pag_dr_slice = slice_im[np.isin(slice_regions, ['DR', 'PAG'])]
        control_slice = slice_im[np.isin(slice_regions, CONTROL_REGIONS)]

        all_rel_fluo[j] = (np.sum(pag_dr_slice[pag_dr_slice > np.percentile(pag_dr_slice, 98)])
                           / np.sum(control_slice[control_slice > np.percentile(control_slice, 98)]))

    rel_fluo = np.max(all_rel_fluo)

    # Plot figures
    f, (ax1, ax2) = plt.subplots(1, 2, dpi=400)
    gr_hist.plot_cslice(-4475/1e6, ax=ax1)
    slice_regions = gr_hist.slice(-4500/1e6, axis=1, volume='value',
                                  region_values=ba.regions.id2acronym(ba.regions.remap(ba.regions.id)))
    plot_regions = np.zeros(slice_regions.shape)
    plot_regions[np.isin(slice_regions, ['DR', 'PAG'])] = 1
    plot_regions[np.isin(slice_regions, CONTROL_REGIONS)] = 2
    ax2.imshow(np.moveaxis(plot_regions, 0, 1))
    ax2.set(title=f'{subject}')

    # Add to dataframe
    expr_df = pd.concat((expr_df, pd.DataFrame(index=[expr_df.shape[0]+1], data={
        'subject': subject,
        'sert-cre': subject_info.loc[subject_info['subject'] == subject, 'sert-cre'].values[0],
        'rel_fluo': rel_fluo})))