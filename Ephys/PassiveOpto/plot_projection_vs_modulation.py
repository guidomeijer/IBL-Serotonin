# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 19:16:16 2022

@author: Guido
"""

from os.path import join
import pandas as pd
from serotonin_functions import paths, remap
from ibllib.atlas import BrainRegions
br = BrainRegions()

fig_path, data_path = paths()
proj_df = pd.read_csv(join(data_path, 'projection_density.csv'))

proj_df['acronym'] = br.id2acronym(proj_df['structure_id'])
proj_df['full_region'] = remap(proj_df['acronym'], combine=True)

proj_df.groupby('full_region').mean()['normalized_projection_volume']


