#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:24:30 2021

@author: guido
"""
import vedo
import pandas as pd
import numpy as np
from iblviewer import atlas_controller
from ibllib.atlas import BrainRegions


def get_children_region_names(acronyms, return_full_name=False):
    br = BrainRegions()
    children_region_names = []
    for i, acronym in enumerate(acronyms):
        try:
            regid = br.id[np.argwhere(br.acronym == acronym)]
            descendants = br.descendants(regid)
            targetlevel = 8
            if sum(descendants.level == targetlevel) == 0:
                if return_full_name:
                    children_region_names.append(descendants.name[-1])
                else:
                    children_region_names.append(descendants.acronym[-1])
            else:
                if return_full_name:
                    children_region_names.append(descendants.name[
                        (descendants.level == targetlevel) & (descendants.id > 0)])
                else:
                    children_region_names.append(descendants.acronym[
                        (descendants.level == targetlevel) & (descendants.id > 0)])
        except IndexError:
            children_region_names.append(acronym)
    if len(children_region_names) == 1:
        return children_region_names[0]
    else:
        return children_region_names


nan_color = [0, 0, 0]
nan_alpha = 0
VMIN = -0.5
VMAX = 0.5
COLOR_MAP = 'seismic'
MIN_REC = 1

# Initialize
resolution = 25  # units = um
mapping = 'Beryl'
controller = atlas_controller.AtlasController()
controller.initialize(resolution, mapping, embed_ui=True, jupyter=False)

# Load in data
df = pd.read_csv('/home/guido/Data/5HT/light_modulated_neurons.csv')

# Duplicate entries for all children
filtered_df = df.groupby('region')['roc_auc'].median()[df.groupby('region').size() > MIN_REC]
filtered_df = filtered_df.groupby(filtered_df.index).first()

# Or for i in range(0, len(df)): which preserves data types
scalars_map = {}
for acronym, value in filtered_df.iteritems():
    if value is None:
        continue
    region_ids, row_ids = controller.model.get_region_and_row_id(acronym)
    if region_ids is None:
        print('Acronym', acronym, 'was not found in Atlas')
        continue
    for r_id in range(len(region_ids)):
        region_id = region_ids[r_id]
        row_id = row_ids[r_id]
        if region_id is None:
            print('Error, could not find acronym (ignoring it)', acronym)
            continue
        if row_id == 0: #or value.isnull().values.any():
            # We ignore void acronym and nan values
            continue
        scalars_map[int(row_id)] = value

rgb = []
alpha = []
for r_id in range(controller.model.atlas.regions.id.size):
    rand_val = np.random.uniform(0, 0.35)
    rgb.append([r_id, np.array([rand_val]*3) + nan_color])
    a = nan_alpha if r_id > 0 else 0.0
    alpha.append([r_id, a])

values = sorted(scalars_map.values())

min_p = VMIN
max_p = VMAX
rng_p = max_p - min_p
#cmap = vedo.colorMap(values, cmap_name, min_p, max_p)
for row_id in scalars_map:
    value = scalars_map[row_id]
    rgb[row_id][1] = list(vedo.colorMap(value, COLOR_MAP, min_p, max_p))
    alpha[row_id] = [row_id, 1.0]


controller.add_transfer_function(scalars_map, rgb, alpha, COLOR_MAP, make_current=False)
controller.render()