#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:43:30 2022
By: Guido Meijer
"""

import sys
sys.path.append('/home/guido/Repositories/ClearMap2')

from ClearMap.Environment import *

#directories and files
directory = '/home/guido/Histology/ZFM-02600'

expression_raw      = 'STD_ds_ZFM-02600_GR.tif'
expression_auto     = 'STD_ds_ZFM-02600_RD.tif'

ws = wsp.Workspace('CellMap', directory=directory);
ws.update(raw=expression_raw, autofluorescence=expression_auto)
ws.debug = False

resources_directory = settings.resources_path

ws.info()

#init atals and reference files
annotation_file, reference_file, distance_file=ano.prepare_annotation_files(
    slicing=(slice(None),slice(None),slice(0,256)), orientation=(1,-2,3),
    overwrite=False, verbose=True);

#alignment parameter files
align_channels_affine_file   = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_affine_file  = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline.txt')

