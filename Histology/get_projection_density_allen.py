# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 19:09:19 2022

@author: Guido
"""

import pandas as pd
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

# The manifest file is a simple JSON file that keeps track of all of
# the data that has already been downloaded onto the hard drives.
# If you supply a relative path, it is assumed to be relative to your
# current working directory.
mcc = MouseConnectivityCache()

structure_tree = mcc.get_structure_tree()
isocortex = structure_tree.get_structures_by_name(['Isocortex'])[0]

# find wild-type injections into primary visual area
dr = structure_tree.get_structures_by_acronym(['DR'])[0]
dr_experiments = mcc.get_experiments(injection_structure_ids=[dr['id']])

print("%d DR experiments" % len(dr_experiments))

structure_unionizes = mcc.get_structure_unionizes([ e['id'] for e in dr_experiments ], 
                                                  is_injection=False,
                                                  include_descendants=True)

# Save to disk
structure_unionizes.to_csv('C:\\Users\\guido\\Repositories\\IBL-Serotonin\\Data\\projection_density.csv')
