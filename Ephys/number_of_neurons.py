# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 12:14:41 2023

@author: Guido
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from serotonin_functions import paths, figure_style, combine_regions, load_subjects

# Paths
fig_path, save_path = paths()

# Load in results
awake_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
anes_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons_anesthesia.csv'))
all_neurons = pd.concat((awake_neurons, anes_neurons))

# Per region
all_neurons['full_region'] = combine_regions(all_neurons['region'])
per_region = all_neurons.groupby('full_region').size().to_frame()
per_mouse = all_neurons.groupby(['subject', 'full_region']).size().reset_index()
per_region['mean'] = per_mouse.groupby('full_region').mean(numeric_only=True).round().astype(int)
per_region['sem'] = per_mouse.groupby('full_region').sem(numeric_only=True).round().astype(int)
per_region = per_region.rename(columns={0: 'total'})
per_region = per_region.sort_values('total', ascending=False)
print(f'\n{all_neurons.shape[0]} neurons in {len(np.unique(all_neurons["subject"]))} mice')
print(per_region)
