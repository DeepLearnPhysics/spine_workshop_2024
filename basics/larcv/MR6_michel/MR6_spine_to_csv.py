import sys

SOFTWARE_DIR = '/global/homes/s/seschwar/spine_workshop/spine/' # Change this path to your software install

args = sys.argv[1:]
if args[0] != '-file':
    -print('Error')
# print(len(args))
DATA_PATH = args[1]
quick_name = DATA_PATH.split('MLRECO_ANALYSIS/')[-1][8:-11]
# DATA_DIR = '/global/cfs/cdirs/dune/users/drielsma/spine_workshop/reco/' # Change this path if you are not on SDF (see main README)

# Set software directory
sys.path.append(SOFTWARE_DIR)

import numpy as np
import math
import pandas as pd
from collections import OrderedDict

from scipy.spatial.distance import cdist

import yaml

# DATA_PATH = '/pscratch/sd/d/dunepr/output/MiniRun6/run-spine/MiniRun6_1E19_RHC.spine/MLRECO_ANALYSIS/*/MiniRun6_1E19_RHC.spine.*.SPINE.hdf5'
print(DATA_PATH)

cfg = '''
# Load HDF5 files
io:
  reader:
    name: hdf5
    file_keys: DATA_PATH
    
# Build reconstruction output representations
build:
  mode: both
  units: cm
  fragments: false
  particles: true
  interactions: true'''.replace('DATA_PATH',DATA_PATH)

cfg = yaml.safe_load(cfg)

from spine.driver import Driver

driver = Driver(cfg)

from spine.utils.globals import SHAPE_LABELS, MICHL_SHP, TRACK_SHP

# Selection parameters
muon_min_voxel_count=30
attach_threshold = 3
match_threshold = 0.5

def fill_true_michels(it, truth_particles, true_michels):
    
    # it: entry number, iteration
    # truth_particles: truth_particles
    # true_michels: output list of ordered dictionary of true michels
    
    for tp in truth_particles:
        if tp.shape != MICHL_SHP: continue
        is_contained = tp.is_contained                                                                                           
        michl_start = tp.start_point
        int_id = tp.interaction_id
        is_attached = False
        
        d = math.inf
        for tp2 in truth_particles:
            if tp2.interaction_id != int_id: continue
            if tp2.shape != TRACK_SHP: continue
            if tp2.size < muon_min_voxel_count: continue

            muon_end = tp2.end_point
            distances = cdist(tp.points, [muon_end])
            d = np.min(distances)
            #d = min(d, np.linalg.norm(michl_start-muon_end))
            if d < attach_threshold:
                is_attached = True
                true_mu = tp2
                break
                
        true_michels.append(OrderedDict({
                    'index': it,
                    'id': tp.id,
                    'attach_dist': d,
                    'is_attached': is_attached,
                    'is_contained': is_contained,
                    'is_matched': tp.is_matched,
                    'match_ids': tp.match_ids,
                    'match_overlaps': tp.match_overlaps,
                    'ke': tp.ke,
                    'parent_file': DATA_PATH
                }))
'''
TT TF FT voxel level / energy sums
R xxxxoo
T oxxxxo

buggy bugs (True KE??? Why is ke in true_michel.csv not matched to true_ke in reco_michel.csv?)

failure modes?
angle
pion/muon
energy dependence?

data vs sim (and with reco)
'''
                 
def fill_michels(it, reco_particles, truth_particles, michels):
    
    # it: entry number, iteration
    # michels: output list of ordered dictionary of michels

    for p in reco_particles:
        muon = None
        if p.shape != MICHL_SHP: continue
        is_contained = p.is_contained
        michl_start = p.start_point
        int_id = p.interaction_id
        is_attached = False

        d = math.inf
        for p2 in reco_particles:
            if p2.interaction_id != int_id: continue
            if p2.shape != TRACK_SHP: continue
            if p2.size < muon_min_voxel_count: continue

            muon_end = p2.end_point
            distances = cdist(p.points, [muon_end])
            d = np.min(distances)
            #d = min(d, np.linalg.norm(michl_start-muon_end))
            if d < attach_threshold:
                is_attached = True
                muon = p2
                break

        # Record candidate Michel                                                                                                                                       
        update_dict={
                'index': it,
                'id': p.id,
                "attach_dist": d,
                "is_attached": is_attached,
                "is_contained": is_contained,
                'is_matched': p.is_matched,
                'match_ids': p.match_ids,
                'match_overlaps': p.match_overlaps,
                "pred_num_pix": p.size,
                "pred_sum_pix": p.depositions.sum(), 
                "true_num_pix": -1,
                "true_sum_pix": -1,
                'ke': p.ke,
                'true_ke': -1,
                'parent_file': DATA_PATH
                }
        
        # Finding matched true michel
        for i in range(len(p.match_ids)):
            if p.match_overlaps[i]>match_threshold: # match_threshold is set to 0.5
                true_id = p.match_ids[i]
                for tp in truth_particles:
                    if tp.id == true_id:
                        m = tp
                        update_dict.update({
                            "true_num_pix": m.size,
                            "true_sum_pix": m.depositions.sum(),
                            "true_ke": m.ke,
                        })

        michels.append(OrderedDict(update_dict))

michels, true_michels = [], []

# from tqdm import tqdm
n_samples = len(driver)
# for iteration in tqdm(range(n_samples)):
for iteration in range(n_samples):
    data = driver.process(entry=iteration)
    
    reco_particles     = data['reco_particles']
    truth_particles    = data['truth_particles'] 

    #selected_michels = get_michels(reco_particles)
    fill_true_michels(iteration, truth_particles, true_michels)
    fill_michels(iteration, reco_particles, truth_particles, michels)#, matched_t2r)

    #fill_michels(michels_it, true_michels_it, matched_r2t, matched_t2r)

michels_df = pd.DataFrame([michels[i] for i, j in enumerate(michels)])
true_michels_df = pd.DataFrame([true_michels[i] for i, j in enumerate(true_michels)])
# -print('michels_csv/'+quick_name+'.csv','true_michels_csv/'+quick_name+'.csv')
# -print('michels_csv/'+quick_name+'.csv')
michels_df.to_csv('reco_michels_csv/'+quick_name+'.csv')#michels_2x2_full.csv')
true_michels_df.to_csv('true_michels_csv/'+quick_name+'.csv')#_2x2_full.csv')