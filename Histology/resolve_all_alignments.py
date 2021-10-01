#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:06:43 2021
By: Guido Meijer
"""

from ibllib.qc.alignment_qc import AlignmentQC
from one.api import ONE
one = ONE()

all_ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,serotonin_inference,'
                        'session__qc__lt,50')
for i, ins in enumerate(all_ins):
    ins_id = ins['id']
    print(f'Resolving insertion {ins_id} [{i+1} of {len(all_ins)}]')
    traj = one.alyx.rest('trajectories', 'list', probe_insertion=ins_id, provenance='Ephys aligned histology track')[0]
    alignment_keys = traj['json'].keys()
    if len(alignment_keys) == 0:
        print('No alignement found')
        continue
    elif len(alignment_keys) > 1:
        print('More than one alignment found!')
        continue
    align_qc = AlignmentQC(ins_id, one=one)
    align_qc.resolve_manual(list(alignment_keys)[0])

