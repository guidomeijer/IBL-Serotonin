#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 18:27:31 2022
By: Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from serotonin_functions import paths, figure_style

# Get paths
fig_path, save_path = paths()

# Load in data
updown_df = pd.read_csv(join(save_path, 'up_down_state_transitions.csv'))

# Get average over regions per animal
per_animal_df = updown_df.groupby(['subject', 'region', 'time']).mean(numeric_only=True).reset_index()

# Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot()
