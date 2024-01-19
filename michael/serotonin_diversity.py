from one.api import ONE
from brainbox.io.one import load_channel_locations 
from brainbox.processing import bincount2D
from ibllib.atlas import regions_from_allen_csv
import ibllib.atlas as atlas

import numpy as np
from pathlib import Path
from collections import Counter
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr, spearmanr, percentileofscore, zscore
from copy import deepcopy
import pandas as pd
import random
 
import itertools
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy import stats, signal
from scipy.signal import hilbert
import random, math
import csv

T_BIN = 0.004  # time bin size in seconds

fname = '/home/mic/GuidoSerotonin/IBL-Serotonin/serotonin_functions.py'
exec(compile(open(fname, "rb").read(),fname, 'exec'))

# segment length: 1000 (4 sec at 250 Hz)

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or 
    math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def get_acronyms_per_insertion(eid, probe):

    T_BIN = 0.5
    one = ONE()                  
                     
    # bin spikes
    spikes = one.load_object(eid, 'spikes', collection=f'alf/{probe}',
                             attribute=['times',
                                        'clusters'])

    R, times, Clusters = bincount2D(spikes['times'],
                                    spikes['clusters'], T_BIN)

    clusters = one.load_object(eid, 'clusters', collection=f'alf/{probe}',
                               attribute=['channels'])                     
                     
    # Get for each cluster the location acronym        
    cluster_chans = clusters['channels'][Clusters]
    els = load_channel_locations(eid, one=one)   
    acronyms = els[probe]['acronym'][cluster_chans]

    return acronyms


def bin_neural(eid, double=True, probe=None, reg=None):
    '''
    bin neural activity; combine probes or pick single region
    ''' 
    one = ONE()     
    
    if probe is not None:
        double = False
    
    
    if double:
        print('combining data from both probes')
        sks = []
        dsets = one.list_datasets(eid)
        r = [x.split('/') for x in dsets if 'probe' in x]
        rr = [item for sublist in r for item in sublist
              if 'probe' in item and '.' not in item]
                                
        for probe in list(Counter(rr)): 
            spikes = one.load_object(eid, 'spikes', collection=f'alf/{probe}',
                                     attribute=['times','clusters'])            
            sks.append(spikes)
         
        # add max cluster of p0 to p1, then concat, sort 
        max_cl0 = max(sks[0]['clusters'])
        sks[1]['clusters'] = sks[1]['clusters'] + max_cl0
         
        times_both = np.concatenate([sks[0]['times'],sks[1]['times']])
        clusters_both = np.concatenate([sks[0]['clusters'],sks[1]['clusters']])
        
        t_sorted = np.sort(times_both)
        c_ordered = clusters_both[np.argsort(times_both)] 
        
        print('binning data')
        R, times, _ = bincount2D(t_sorted, c_ordered, T_BIN)  

        D = R.T            
        
    else:    
        print('single probe') 
        spikes = one.load_object(eid, 'spikes', collection=f'alf/{probe}',
                                 attribute=['times','clusters'])   
          
     
        # bin spikes
        R, times, Clusters = bincount2D(
            spikes['times'], spikes['clusters'], T_BIN)
       
        D = R.T       
        
        if reg is not None:
            
            # Get for each cluster the location x,y,z
            clusters = one.load_object(eid, 'clusters', collection=f'alf/{probe}',
                                       attribute=['channels'])    
            cluster_chans = clusters['channels'][Clusters]      
            els = load_channel_locations(eid, one=one)   
            acronyms = els[probe]['acronym'][cluster_chans]       
            m_ask = acronyms == reg
            D = D[:,m_ask]

    return D, times 

    
    
def cpr(string):
    '''
    Lempel-Ziv-Welch compression of binary input string, e.g. string='0010101'. It outputs the size of the dictionary of binary words.
    '''
    d = {'0':'0','1':'1'} 
    w = ''
    i = 1
    for c in string:
        wc = w + c
        if wc in d:
            w = wc
        else:
            d[wc] = wc
            w = c
        i+=1
    return len(d) 
    
    
def LZs(x):
 
    '''
    Lempel ziv complexity of single timeseries
    '''

    #this differes from Sitt et al as they use 32 bins, not just 2. 
    co = len(x)
    x = signal.detrend((x-np.mean(x))/np.std(x), axis=0)
    s = ''
    r = abs(hilbert(x))
    th = np.mean(r)

    for j in range(co):
        if r[j] > th:
            s += '1'
        else:
            s += '0'

    M = list(s)
    random.shuffle(M)
    w = ''
    for i in range(len(M)):
        w += M[i]

    return cpr(s)/float(cpr(w))   
    
    
def LZs_(eid,probe,reg=None):

    D, times = bin_neural(eid, double=True, probe=probe, reg=reg)  
    ot = load_opto_times(eid)
    
    one = ONE()
    trials = one.load_object(eid, 'trials')
    t_start = trials['intervals'][-1][-1] 
    idx = find_nearest(times, t_start + 10)
    
    D = D.mean(axis=1)  #firing rate per region or whole probe
    
    l = len(times[idx:])
    ll = 2000 # number of observations per segment
    lz_times = []
    lz_scores = [] 
    
    for seg in range(l//ll):
        ts = times[idx + seg * ll: idx + (seg + 1) * ll]    
        rs = LZs(D[idx + seg * ll: idx + (seg + 1) * ll])
        lz_times.append(ts[0])
        lz_scores.append(rs) 
    
    return D, times, ot, lz_times, lz_scores, t_start + 10

  
'''
batch processing
'''  
  
def get_LZ_REGION():
    from one.api import ONE
    one = ONE()
    all_sess = one.alyx.rest('insertions', 'list',
                           django='session__project__name__icontains,'
                           'serotonin_inference,session__qc__lt,50')
                             
    inserts = [[s['session'],s['name']] for s in all_sess]     
    plt.ioff()

    R = {}
    for ins in inserts:
        eid, probe = ins
        acronyms = get_acronyms_per_insertion(eid, probe)
        for reg in Counter(acronyms):
            if Counter(acronyms)[reg] > 30:
                ###LZ =  
     
                R[f'{eid}_{probe}_{reg}'] = LZ   
    print(R)    
    np.save('/home/mic/paper-brain-wide-map/manifold_analysis/'
            f'manifold_ps/per_region.npy', R, allow_pickle=True)       


'''
plotting
'''    

def plot_passive(eid, probe):

    #acs = get_acronyms_per_insertion(eid, probe)
    
    reader = csv.reader(open('/home/mic/GuidoSerotonin/subjects.csv', 'r'))
    d = {}
    for row in reader:
       q = list(row)
       d[q[0]] = q[1]

    one = ONE()
    ref = one.eid2ref(eid)
    

    D, times, ot, lz_times, lz_scores, t_start = LZs_(eid,probe)
    
    
    plt.plot(times,D,linewidth=0.1)
    
    ax = plt.gca() 
    for t in ot:
        ax.axvline(x=t,color='r',linestyle='--')
    
       
    ax.axvline(x=t,color='r',linestyle='--',label='opto stim')
    plt.ylabel('spike rate averaged across probe')
    plt.title(f"{eid}, {probe}, sert-cre: {d[ref['subject']]} \n"
              f"{ref['subject']}, {ref['date']}")
    plt.xlabel('time [sec]')    
    plt.legend()  
    ax = plt.gca()
    ax2 = ax.twinx()
    
    ax2.plot(lz_times,lz_scores, c='k',label='LZ',linewidth=2)
    ax2.set_ylabel('LZ score for 4 sec segments')
    plt.legend()
    plt.xlim(left=t_start)
    plt.tight_layout()
    plt.savefig(f"/home/mic/GuidoSerotonin/plots/"
                f"{eid}_{probe}_{ref['subject']}_{d[ref['subject']]}")    
    plt.close() 




