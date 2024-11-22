# %% [markdown]
# python3 bin/run_flow2supera.py -o test_sim.root -n 1000 -c 2x2 /global/cfs/cdirs/dune/www/data/2x2/simulation/productions/MiniRun5_1E19_RHC/MiniRun5_1E19_RHC.flow.beta2a/FLOW/0000000/MiniRun5_1E19_RHC.flow.0000000.FLOW.hdf5
# <!-- /global/cfs/cdirs/dune/www/data/2x2/simulation/productions/MiniRun5_1E19_RHC/MiniRun5_1E19_RHC.mlreco_analysis.beta2a/MLRECO_ANA/0000000/MiniRun5_1E19_RHC.mlreco_analysis.0000000.MLRECO_ANA.hdf5 -->

# %%
import numpy as np
import math
import pandas as pd
from collections import OrderedDict

from scipy.spatial.distance import cdist

# %%
import matplotlib
import matplotlib.pyplot as plt
import seaborn
seaborn.set(rc={
    'figure.figsize':(15, 10),
})
seaborn.set_context('talk')

# %% [markdown]
# ## II. Setup
# 
# We first need to set the working environment and the path to the dataset.

# %%
import sys

SOFTWARE_DIR = '/global/homes/s/seschwar/spine_workshop/spine/' # Change this path to your software install

DATA_DIR = '/global/cfs/cdirs/dune/users/drielsma/spine_workshop/reco/' # Change this path if you are not on SDF (see main README)

# Set software directory
sys.path.append(SOFTWARE_DIR)

# %% [markdown]
# Also, let's import some libraries

# %% [markdown]
# Now pass the analysis configuration

# %%
import yaml
# DATA_PATH = '/global/cfs/cdirs/dune/www/data/2x2/simulation/productions/MiniRun4.5_1E19_RHC/inputs_beta3/MiniRun4.5_1E19_RHC.mlreco_analysis/MLRECO_ANA/0000000/MiniRun4.5_1E19_RHC.mlreco_analysis.0000000.MLRECO_ANA.hdf5'
# DATA_PATH = '/global/cfs/cdirs/dune/www/data/2x2/simulation/productions/MiniRun5_1E19_RHC/MiniRun5_1E19_RHC.mlreco_analysis.beta2a/MLRECO_ANA/0000000/MiniRun5_1E19_RHC.mlreco_analysis.0000001.MLRECO_ANA.hdf5'
# DATA_PATH = '/global/cfs/cdirs/dune/www/data/2x2/simulation/productions/MiniRun5_1E19_RHC/MiniRun5_1E19_RHC.mlreco_analysis.beta2a/MLRECO_ANA/0000000/MiniRun5_1E19_RHC.mlreco_analysis.0000000.MLRECO_ANA.hdf5'
# DATA_PATH = '/global/homes/s/seschwar/spine_workshop/spine_workshop_2024/basics/larcv/test.h5'
# DATA_PATH = '/global/cfs/cdirs/dune/www/data/2x2/simulation/productions/MiniRun5_1E19_RHC/MiniRun5_1E19_RHC.flow.beta2a/FLOW/0000000/MiniRun5_1E19_RHC.flow.0000000.FLOW.hdf5'

# cfg = '''
# # Base configuration
# base:
#   world_size: 1
#   iterations: -1
#   seed: 0
#   dtype: float32
#   unwrap: true
#   log_dir: logs
#   prefix_log: true
#   overwrite_log: true
#   log_step: 1

# # IO configuration
# io:
#   loader:
#     batch_size: 64
#     shuffle: false
#     num_workers: 8
#     collate_fn: all
#     dataset:
#       name: larcv
#       file_keys: DATA_PATH
#       schema:
#         data:
#           parser: sparse3d
#           sparse_event: sparse3d_pcluster
#         seg_label:
#           parser: sparse3d
#           sparse_event: sparse3d_pcluster_semantics
#         ppn_label:
#           parser: particle_points
#           sparse_event: sparse3d_pcluster
#           particle_event: particle_pcluster
#           include_point_tagging: false
#         clust_label:
#           parser: cluster3d
#           cluster_event: cluster3d_pcluster
#           particle_event: particle_pcluster
#           neutrino_event: neutrino_mc_truth
#           sparse_semantics_event: sparse3d_pcluster_semantics
#           add_particle_info: true
#           clean_data: true
#         coord_label:
#           parser: particle_coords
#           particle_event: particle_pcluster
#           cluster_event: cluster3d_pcluster
#         graph_label:
#           parser: particle_graph
#           particle_event: particle_pcluster
#         particles:
#           parser: particle
#           particle_event: particle_pcluster
#           neutrino_event: neutrino_mc_truth
#           cluster_event: cluster3d_pcluster
#           skip_empty: true
#         neutrinos:
#           parser: neutrino
#           neutrino_event: neutrino_mc_truth
#           cluster_event: cluster3d_pcluster
#         meta:
#           parser: meta
#           sparse_event: sparse3d_pcluster
#         run_info:
#           parser: run_info
#           sparse_event: sparse3d_pcluster
#         trigger:
#           parser: trigger
#           trigger_event: trigger_base

#   writer:
#     name: hdf5
#     file_name: null
#     overwrite: true
#     keys:
#       - run_info
#       - meta
#       - trigger
#       - points
#       - points_label
#       - depositions
#       - depositions_label
#       - reco_particles
#       - truth_particles
#       - reco_interactions
#       - truth_interactions

# # Model configuration
# model:
#   name: full_chain
#   weight_path: /global/cfs/cdirs/dune/users/drielsma/weights/2x2_240819_snapshot.ckpt
#   to_numpy: true

#   network_input:
#     data: data
#     seg_label: seg_label
#     clust_label: clust_label

#   loss_input:
#     seg_label: seg_label
#     ppn_label: ppn_label
#     clust_label: clust_label
#     coord_label: coord_label

#   modules:
#     # General chain configuration
#     chain:
#       deghosting: null
#       charge_rescaling: null
#       segmentation: uresnet
#       point_proposal: ppn
#       fragmentation: graph_spice
#       shower_aggregation: grappa
#       shower_primary: grappa
#       track_aggregation: grappa
#       particle_aggregation: null
#       inter_aggregation: grappa
#       particle_identification: grappa
#       primary_identification: grappa
#       orientation_identification: grappa
#       calibration: null

#     # Semantic segmentation + point proposal
#     uresnet_ppn:
#       uresnet:
#         num_input: 1
#         num_classes: 5
#         filters: 32
#         depth: 5
#         reps: 2
#         allow_bias: false
#         activation:
#           name: lrelu
#           negative_slope: 0.33
#         norm_layer:
#           name: batch_norm
#           eps: 0.0001
#           momentum: 0.01
  
#       ppn:
#         classify_endpoints: false
  
#     uresnet_ppn_loss:
#       uresnet_loss:
#         balance_loss: false
  
#       ppn_loss:
#         mask_loss: CE
#         resolution: 5.0

#     # Dense clustering
#     graph_spice:
#       shapes: [shower, track, michel, delta]
#       use_raw_features: true
#       invert: true
#       make_clusters: true
#       embedder:
#         spatial_embedding_dim: 3
#         feature_embedding_dim: 16
#         occupancy_mode: softplus
#         covariance_mode: softplus
#         uresnet:
#           num_input: 4 # 1 feature + 3 normalized coords
#           filters: 32
#           input_kernel: 5
#           depth: 5
#           reps: 2
#           spatial_size: 320
#           allow_bias: false
#           activation:
#             name: lrelu
#             negative_slope: 0.33
#           norm_layer:
#             name: batch_norm
#             eps: 0.0001
#             momentum: 0.01
#       kernel:
#         name: bilinear
#         num_features: 32
#       constructor:
#         edge_threshold: 0.1
#         min_size: 3
#         label_edges: true
#         graph:
#           name: radius
#           r: 1.9
#         orphan:
#           mode: radius
#           radius: 1.9
#           iterate: true
#           assign_all: true

#     graph_spice_loss:
#       name: edge
#       loss: binary_log_dice_ce

#     # Shower fragment aggregation + shower primary identification
#     grappa_shower:
#       nodes:
#         source: cluster
#         shapes: [shower, michel, delta]
#         min_size: -1
#         make_groups: true
#         grouping_method: score
#       graph:
#         name: complete
#         max_length: [500, 0, 500, 500, 0, 0, 0, 25, 0, 25]
#         dist_algorithm: recursive
#       node_encoder:
#         name: geo
#         use_numpy: true
#         add_value: true
#         add_shape: true
#         add_points: true
#         add_local_dirs: true
#         dir_max_dist: 5
#         add_local_dedxs: true
#         dedx_max_dist: 5
#       edge_encoder:
#         name: geo
#         use_numpy: true
#       gnn_model:
#         name: meta
#         node_feats: 33 # 16 (geo) + 3 (extra) + 6 (points) + 6 (directions) + 2 (local dedxs)
#         edge_feats: 19
#         node_pred: 2
#         edge_pred: 2
#         edge_layer:
#           name: mlp
#           mlp:
#             depth: 3
#             width: 64
#             activation:
#               name: lrelu
#               negative_slope: 0.1
#             normalization: batch_norm
#         node_layer:
#           name: mlp
#           reduction: max
#           attention: false
#           message_mlp:
#             depth: 3
#             width: 64
#             activation:
#               name: lrelu
#               negative_slope: 0.1
#             normalization: batch_norm
#           aggr_mlp:
#             depth: 3
#             width: 64
#             activation:
#               name: lrelu
#               negative_slope: 0.1
#             normalization: batch_norm

#     grappa_shower_loss:
#       node_loss:
#         name: shower_primary
#         high_purity: true
#         use_group_pred: true
#       edge_loss:
#         name: channel
#         target: group
#         high_purity: true

#     # Track aggregation
#     grappa_track:
#       nodes:
#         source: cluster
#         shapes: [track]
#         min_size: -1
#         make_groups: true
#         grouping_method: score
#       graph:
#         name: complete
#         max_length: 100
#         dist_algorithm: recursive
#       node_encoder:
#         name: geo
#         use_numpy: true
#         add_value: true
#         add_shape: false
#         add_points: true
#         add_local_dirs: true
#         dir_max_dist: 5
#         add_local_dedxs: true
#         dedx_max_dist: 5
#       edge_encoder:
#         name: geo
#         use_numpy: true
#       gnn_model:
#         name: meta
#         node_feats: 32 # 16 (geo) + 2 (extra) + 6 (points) + 6 (directions) + 2 (local dedxs)
#         edge_feats: 19
#         edge_pred: 2
#         edge_layer:
#           name: mlp
#           mlp:
#             depth: 3
#             width: 64
#             activation:
#               name: lrelu
#               negative_slope: 0.1
#             normalization: batch_norm
#         node_layer:
#           name: mlp
#           reduction: max
#           attention: false
#           message_mlp:
#             depth: 3
#             width: 64
#             activation:
#               name: lrelu
#               negative_slope: 0.1
#             normalization: batch_norm
#           aggr_mlp:
#             depth: 3
#             width: 64
#             activation:
#               name: lrelu
#               negative_slope: 0.1
#             normalization: batch_norm

#     grappa_track_loss:
#       edge_loss:
#         name: channel
#         target: group

#     # Interaction aggregation, PID, primary, orientation
#     grappa_inter:
#       nodes:
#         source: group
#         shapes: [shower, track, michel, delta]
#         min_size: -1
#         make_groups: true
#       graph:
#         name: complete
#         max_length: [500, 500, 0, 0, 25, 25, 25, 0, 0, 0]
#         dist_algorithm: recursive
#       node_encoder:
#         name: geo
#         use_numpy: true
#         add_value: true
#         add_shape: true
#         add_points: true
#         add_local_dirs: true
#         dir_max_dist: 5
#         add_local_dedxs: true
#         dedx_max_dist: 5
#       edge_encoder:
#         name: geo
#         use_numpy: true
#       gnn_model:
#         name: meta
#         node_feats: 33 # 16 (geo) + 3 (extra) + 6 (points) + 6 (directions) + 2 (local dedxs)
#         edge_feats: 19
#         node_pred:
#           type: 6
#           primary: 2
#           orient: 2
#           #momentum: 1
#           #vertex: 5
#         edge_pred: 2
#         edge_layer:
#           name: mlp
#           mlp:
#             depth: 3
#             width: 128
#             activation:
#               name: lrelu
#               negative_slope: 0.1
#             normalization: batch_norm
#         node_layer:
#           name: mlp
#           reduction: max
#           attention: false
#           message_mlp:
#             depth: 3
#             width: 128
#             activation:
#               name: lrelu
#               negative_slope: 0.1
#             normalization: batch_norm
#           aggr_mlp:
#             depth: 3
#             width: 128
#             activation:
#               name: lrelu
#               negative_slope: 0.1
#             normalization: batch_norm

#     grappa_inter_loss:
#       node_loss:
#         type:
#           name: class
#           target: pid
#           loss: ce
#           balance_loss: true
#         primary:
#           name: class
#           target: inter_primary
#           loss: ce
#           balance_loss: true
#         orient:
#           name: orient
#           loss: ce
#         #momentum:
#         #  name: reg
#         #  target: momentum
#         #  loss: berhu
#         #vertex:
#         #  name: vertex
#         #  primary_loss: ce
#         #  balance_primary_loss: true
#         #  regression_loss: mse
#         #  only_contained: true
#         #  normalize_positions: true
#         #  use_anchor_points: true
#         #  return_vertex_labels: true
#         #  detector: icarus
#       edge_loss:
#         name: channel
#         target: interaction

# # Build output representations
# build:
#   mode: both
#   units: cm
#   fragments: false
#   particles: true
#   interactions: true
  
# # Run post-processors
# post:
#   shape_logic:
#     enforce_pid: true
#     enforce_primary: true
#     priority: 3
#   #particle_threshold:
#   #  track_pid_thresholds:
#   #    4: 0.85
#   #    2: 0.1
#   #    3: 0.5
#   #    5: 0.0
#   #  shower_pid_thresholds:
#   #    0: 0.5 
#   #    1: 0.0
#   #  primary_threshold: 0.1
#   #  priority: 2
#   #track_extrema:
#   #  method: gradient
#   #  priority: 2
#   direction:
#     obj_type: particle
#     optimize: true
#     run_mode: both
#     priority: 1
#   calo_ke:
#     run_mode: reco
#     scaling: 1.
#     shower_fudge: 1/0.83
#     priority: 1
#   csda_ke:
#     run_mode: reco
#     tracking_mode: step_next
#     segment_length: 5.0
#     priority: 1
#   mcs_ke:
#     run_mode: reco
#     tracking_mode: bin_pca
#     segment_length: 5.0
#     priority: 1
#   topology_threshold:
#     ke_thresholds:
#       4: 50
#       default: 25
#   vertex:
#     use_primaries: true
#     update_primaries: false
#     priority: 1
#   containment:
#     detector: 2x2
#     margin: 5.0
#     mode: detector
#   fiducial:
#     detector: 2x2
#     margin: 15.0
#     mode: detector
#   children_count:
#     mode: shape
#   match:
#     match_mode: both
#     ghost: false
#     fragment: false
#     particle: true
#     interaction: true
# '''.replace('DATA_PATH', DATA_PATH)

# DATA_PATH = '/pscratch/sd/d/dunepr/output/MiniRun6/run-spine/MiniRun6_1E19_RHC.spine/MLRECO_ANALYSIS/*/MiniRun6_1E19_RHC.spine.*.SPINE.hdf5'
DATA_PATH = '/pscratch/sd/d/dunepr/output/MiniRun6/run-spine/MiniRun6_1E19_RHC.spine/MLRECO_ANALYSIS/0000000/MiniRun6_1E19_RHC.spine.0000736.SPINE.hdf5'
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


# %% [markdown]
# Now initialize the driver with this configuration

# %%
from spine.driver import Driver

driver = Driver(cfg)

# %% [markdown]
# Let's get the first entry

# %%
ENTRY = 50
# data = driver.process()
data = driver.process(entry=ENTRY)

# %% [markdown]
# Let's get reconstructed/truth particles

# %%
reco_particles     = data['reco_particles']
truth_particles    = data['truth_particles']
data.keys()

# %% [markdown]
# Let's import michel shape and track shape from spine.utils.globals

# %%
from spine.utils.globals import SHAPE_LABELS, MICHL_SHP, TRACK_SHP, PID_LABELS
print(PID_LABELS)

# %%
print(SHAPE_LABELS)

# %%
print(MICHL_SHP, TRACK_SHP)

# %% [markdown]
# Let's see if this entry has Michels

# %%
print('Number of reconstructed Michel:', np.sum(np.array([p.shape for p in reco_particles]) == MICHL_SHP))
print('Number of true Michel:', np.sum(np.array([p.shape for p in truth_particles]) == MICHL_SHP))

# %% [markdown]
# ## III. Selecting Michel Electrons
# 
# Let's draw Michels in this entry.

# %%
from spine.vis.out import Drawer
drawer = Drawer(data, detector='2x2') # Try to replace none with 'icarus', 'sbnd' or '2x2'!

# With 'shape' option, the color indicdates particle shape
fig = drawer.get('particles', ['shape','pid'], draw_end_points=True, draw_vertices=True)

# %%
for reco_p, truth_p in data['particle_matches_t2r']:
    print('\n', reco_p)
    print(truth_p)

# %%
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    width=1100,
    height=500
)
fig.show()
# fig.write_html('sample.html')

#  %%
def eventDisplayer(file_path='/pscratch/sd/d/dunepr/output/MiniRun6/run-spine/MiniRun6_1E19_RHC.spine/MLRECO_ANALYSIS/0000000/MiniRun6_1E19_RHC.spine.0000013.SPINE.hdf5',event_ind=11,save=False,save_name='',verbose=False):
    DATA_PATH = file_path

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

    ENTRY = event_ind
    # data = driver.process()
    data = driver.process(entry=ENTRY)

    reco_particles     = data['reco_particles']
    truth_particles    = data['truth_particles']
    # data.keys()

    from spine.utils.globals import SHAPE_LABELS, MICHL_SHP, TRACK_SHP, PID_LABELS

    if verbose:
        print('Number of reconstructed Michel:', np.sum(np.array([p.shape for p in reco_particles]) == MICHL_SHP))
        print('Number of true Michel:', np.sum(np.array([p.shape for p in truth_particles]) == MICHL_SHP))

    from spine.vis.out import Drawer
    drawer = Drawer(data, detector='2x2') # Try to replace none with 'icarus', 'sbnd' or '2x2'!

    # With 'shape' option, the color indicdates particle shape
    fig = drawer.get('particles', ['shape','pid'], draw_end_points=True, draw_vertices=True)

    if verbose:
        for reco_p, truth_p in data['particle_matches_t2r']:
            print('\n', reco_p)
            print(truth_p)

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        width=2200,
        height=1100
    )
    if verbose:
        fig.show()
    if save:
        fig.write_html('plots/%s.html'%save_name)
    if verbose:
        print(file_path)
        print(event_ind)
    reco_michels_hs = pd.read_csv('MR6_michel/reco_michels.csv')
    true_michels_hs = pd.read_csv('MR6_michel/true_michels.csv')
    reco_file = reco_michels_hs[reco_michels_hs['parent_file']==file_path]
    true_file = true_michels_hs[true_michels_hs['parent_file']==file_path]
    reco_evt = reco_file[reco_file['index']==event_ind]
    true_evt = true_file[true_file['index']==event_ind]
    return reco_evt,true_evt
# %%
events_of_interest='''0000005	26
0000006	58
0000006	80
0000006	164
0000009	112
0000009	122
0000010	44
0000010	66
0000011	80
0000012	21
0000013	11
0000014	160
0000023	47
0000031	93
0000037	115
0000037	168
0000040	54
0000040	92
0000041	97
0000041	108
0000042	36
0000044	64
0000045	176'''.split('\n')
for file_ind,event_ind in map(str.split,events_of_interest):
    event_ind = int(event_ind)
    short_name = file_ind+'_'+str(event_ind)
    save = True
    reco_evt,true_evt=eventDisplayer(file_path='/pscratch/sd/d/dunepr/output/MiniRun6/run-spine/MiniRun6_1E19_RHC.spine/MLRECO_ANALYSIS/%s/MiniRun6_1E19_RHC.spine.%s.SPINE.hdf5'%(['0000000','0001000'][file_ind[3]=='1'],file_ind),event_ind=event_ind,save=save,save_name=short_name,verbose=False)
#%%
index = 0
file_ind,event_ind = events_of_interest[index].split()
event_ind = int(event_ind)
# file_ind = '0000005'
# event_ind = 26
short_name = file_ind+'_'+str(event_ind)
save = False
# ABOVE LINE VERY IMPORTANT #
verbose = 1
reco_evt,true_evt=eventDisplayer(file_path='/pscratch/sd/d/dunepr/output/MiniRun6/run-spine/MiniRun6_1E19_RHC.spine/MLRECO_ANALYSIS/%s/MiniRun6_1E19_RHC.spine.%s.SPINE.hdf5'%(['0000000','0001000'][file_ind[3]=='1'],file_ind),event_ind=event_ind,save=save,save_name=short_name,verbose=verbose)
print(file_ind)
print(event_ind)
print()
print('true table:')
display(true_evt)
print('reco table:')
display(reco_evt)
# %% [markdown]
# ### Selecting Michel candidates
# **Criteria 1: closeness of Michel and track end point**
# To check if our candidate michel is actually adjacent to a track, we will set a minimum distance threshold (`attached_threshold`) between the Michel electron and the end point of the track within the same interaction.
# 

# %%
# Selection parameters
muon_min_voxel_count=30
attach_threshold = 3
match_threshold = 0.5

# %% [markdown]
# ### Finding true Michels
# 
# Let's find true Michels first. 
# 

# %%
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
                    'ke': tp.ke
                }))

# %%
true_michels = []
fill_true_michels(ENTRY, truth_particles, true_michels)

# %%
true_michels

# %% [markdown]
# ### Finding reco Michels
# 
# For reco Michels, we want to save more information than true_michels.\
# If there exists a reco-to-true-matching over the matching threshold, matched true information is also saved.

# %%
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

# %%
michels = []
michels_it = [part for part in truth_particles if part.shape==MICHL_SHP]
fill_michels(ENTRY, reco_particles, truth_particles, michels)

# %%
michels

# %% [markdown]
# We found a Michel and the matched true Michel!

# %% [markdown]
# ## IV. Repeating with high stat
# 
# It's convinient to save [michels, true_michels] to csv.\
# I ran below 3 cells to generate csvs with 1000 images.

# %%
michels, true_michels = [], []

from tqdm import tqdm
n_samples = len(driver)
for iteration in tqdm(range(n_samples)):
    data = driver.process(entry=iteration)
    
    reco_particles     = data['reco_particles']
    truth_particles    = data['truth_particles'] 

    #selected_michels = get_michels(reco_particles)
    fill_true_michels(iteration, truth_particles, true_michels)
    fill_michels(iteration, reco_particles, truth_particles, michels)#, matched_t2r)

    #fill_michels(michels_it, true_michels_it, matched_r2t, matched_t2r)


# %%
michels_df = pd.DataFrame([michels[i] for i, j in enumerate(michels)])
true_michels_df = pd.DataFrame([true_michels[i] for i, j in enumerate(true_michels)])
reco_michels_hs = michels_df
true_michels_hs = true_michels_df

# %%
# michels_df.to_csv('michels_2x2.csv')
# true_michels_df.to_csv('true_michels_2x2.csv')

# %% [markdown]
# Now let's load from those csvs.

# %%
reco_michels_hs = pd.read_csv('MR6_michel/reco_michels.csv')
true_michels_hs = pd.read_csv('MR6_michel/true_michels.csv')
# print(true_michels_hs.keys())
# true_michels_hs[true_michels_hs['Unnamed: 0.1']==true_michels_hs['Unnamed: 0'][1]]
display(reco_michels_hs['true_sum_pix']/reco_michels_hs['true_ke'])

# %%
# true_michels_hs['ke'][true_michels_hs['ke'] in reco_michels_hs['true_ke']]
inds = [i for i in range(len(reco_michels_hs['true_ke']))if reco_michels_hs['true_ke'][i] in true_michels_hs['ke']]
# print(reco_michels_hs['true_ke'])
# print(reco_michels_hs['true_ke'][785])
# [i for i  in true_michels_hs['attach_dist'] if i in reco_michels_hs['attach_dist']]
inds = [i for i in range(len(true_michels_hs['ke']))if true_michels_hs['ke'][i] in reco_michels_hs['true_ke']]
print(inds)
# true_michels_hs[true_michels_hs['ke']in reco_michels_hs['true_ke']]
fig = plt.figure(figsize=(12,7))


# %% [markdown]
# Let's take a look at 'reco_michels_hs'

# %%
# test = pd.concat([reco_michels_hs,true_michels_df],ignore_index=True)
# test = pd.DataFrame(test)
# print(test.keys())
# print(test['id'][3])
# testa

# %% [markdown]
# We'll apply some cuts on selected Michels and see how the efficiency/purity change.

# %%
# no cut

N_pred = reco_michels_hs.shape[0] # count reconstructed michels
N_matched = np.count_nonzero(reco_michels_hs['true_num_pix'] > -1) # count matched michels
N_true = true_michels_hs.shape[0] # count true michels

print("Number of predicted candidate Michel electrons = ", N_pred)
print("Number of matched predicted candidate Michel electrons = ", N_matched)
print("Number of true Michel electrons = ", N_true)
print("Identification purity = %.2f %%" % (100* N_matched / N_pred))
print("Identification efficiency = %.2f %%" % (100 * N_matched / N_true))


# %%
# 0-th cut
# reco. michel is larger than 20 voxels

michels_hs_0 = reco_michels_hs[(reco_michels_hs['pred_num_pix']>=20)] 
N_pred_0 = michels_hs_0.shape[0]
N_matched_0 = np.count_nonzero(michels_hs_0['true_num_pix'] > -1)
N_true = true_michels_hs.shape[0]
print("Number of predicted candidate Michel electrons = ", N_pred_0)
print("Number of matched predicted candidate Michel electrons = ", N_matched_0)
print("Number of true Michel electrons = ", N_true)
print("Identification purity = %.2f %%" % (100* N_matched_0 / N_pred_0))
print("Identification efficiency = %.2f %%" % (100 * N_matched_0 / N_true))

# %%
# 1-th cut
# reco. michel and true michels are attached to tracks within threshold. 

michels_hs_1 = michels_hs_0[michels_hs_0['is_attached']==True]
N_pred_1 = michels_hs_1.shape[0]
N_matched_1 = np.count_nonzero(michels_hs_1['true_num_pix'] > -1)
N_true_1 = true_michels_hs[true_michels_hs['is_attached']==True].shape[0]
print("Number of predicted candidate Michel electrons = ", N_pred_1)
print("Number of matched predicted candidate Michel electrons = ", N_matched_1)
print("Number of true Michel electrons = ", N_true_1)
print("Identification purity = %.2f %%" % (100* N_matched_1 / N_pred_1))
print("Identification efficiency = %.2f %%" % (100 * N_matched_1 / N_true_1))
print('Caveat: Did not remove R michels matched to cut T michels, therefore # of matches is likely an overcount')

# %%
# 2-nd cut
# reco. michel and true michels are contained

michels_hs_2 = michels_hs_1[michels_hs_1['is_contained']==True]
N_pred_2 = michels_hs_2.shape[0]
N_matched_2 = np.count_nonzero(michels_hs_2['true_num_pix'] > -1)
N_true_2 = true_michels_hs[true_michels_hs['is_attached']==True][true_michels_hs[true_michels_hs['is_attached']==True]['is_contained']==True].shape[0]
print("Number of predicted candidate Michel electrons = ", N_pred_2)
print("Number of matched predicted candidate Michel electrons = ", N_matched_2)
print("Number of true Michel electrons = ", N_true_2)
print("Identification purity = %.2f %%" % (100* N_matched_2 / N_pred_2))
print("Identification efficiency = %.2f %%" % (100 * N_matched_2 / N_true_2))
print('Caveat: Did not remove R michels matched to cut T michels, therefore # of matches is likely an overcount')

# %% [markdown]
# It seems like we achieved high efficiency / high purity selection.\
# Now, let's plot some sanity check plots for the selection.

# %% [markdown]
# ### Predicted vs. true voxel counts

# %%
michels_hs_1 = reco_michels_hs
plt.figure(figsize=(12,7))
plt.hist2d(michels_hs_1['true_num_pix'],
          michels_hs_1['pred_num_pix'],
          bins=[40, 40], range=[[0, 400], [0, 400]], cmap='winter', norm=matplotlib.colors.LogNorm())
plt.colorbar()
xy = np.linspace(0,300,41)
plt.plot(xy, xy)
plt.xlabel('Num of voxels in true Michels')
plt.ylabel('Num of voxels in reconstructed Michels')
plt.gca().set_aspect('equal')
plt.show()

# %% [markdown]
# ### Predicted vs. true deposition sum

# %%
plt.figure(figsize=(12,7))
# true_michels_hs
# reco_michels_hs
# plt1 = reco_michels_hs['pred_sum_pix'][reco_michels_hs['is_matched']*reco_michels_hs['is_contained']]
plt1 = true_michels_hs['ke'][true_michels_hs['is_contained']]#[true_michels_hs['is_contained']]
# print(true_michels_hs['mato'])
print(len([max([0,*map(float,i[1:-1].split())]) for i in true_michels_hs['match_overlaps'][true_michels_hs['is_contained']]]))
# print(np.count_nonzero([max([0,*map(float,i[1:-1].split())])>0.5 for i in true_michels_hs['match_overlaps']]))
plt2 = true_michels_hs['ke'][true_michels_hs['is_contained']][np.array([len(i)>0 and max([0,*map(float,i[1:-1].split())])>0.5 for i in true_michels_hs['match_overlaps'][true_michels_hs['is_contained']]])]
# plt2 = true_michels_hs['ke'][np.max(true_michels_hs['match_overlaps'])>0.5]#[true_michels_hs['is_contained']]
# plt2 = reco_michels_hs['true_sum_pix'][reco_michels_hs['is_contained']*reco_michels_hs['is_matched']]
y1,bins=np.histogram(plt1,bins=100,range=(0,60))
y2,bins=np.histogram(plt2,bins=100,range=(0,60))
plt.subplot(211)
plt.hist(plt1,bins = 100, range = (0,60),alpha=.5,label='All (N=%i)'%np.count_nonzero(plt1),density=0)
plt.hist(plt2,bins = 100, range = (0,60),alpha=.5,label='Matched (N=%i)'%np.count_nonzero(plt2),density=0)
plt.legend()
plt.ylabel('Counts')
plt.subplot(212)
plt.ylabel('Ratio (Matched/All)')
plt.plot((bins[1:]+bins[:-1])/2,y2/y1)
# plt.hist2d(plt1,plt2,range=([0,200],[0,200]),bins=40,cmap='winter', norm=matplotlib.colors.LogNorm())
# plt.colorbar()
# plt.gca().set_aspect('equal')
plt.suptitle('MiniRun6 Contained Michels')
plt.xlabel('ke (MeV)')
plt.show()

# %%

plt.figure(figsize=(12,7))
plt.hist2d(michels_hs_1['true_ke'],
          michels_hs_1['ke'],
          bins=[40, 40], range=[[0, 60], [0, 60]], cmap='winter', norm=matplotlib.colors.LogNorm())
plt.colorbar()
xy = np.linspace(0,60,61)
plt.plot(xy, xy)
plt.xlabel('True deposition sum [MeV]')
plt.ylabel('Reconstructed deposition sum [MeV]')
plt.gca().set_aspect('equal')
plt.show()

# %% [markdown]
# ## Exercises
# 1. Identify sources of inneficiency\
# Find which entries have true Michels that are not reconstructed\
# Visualize these entries in an event display\
# Catalogue what is going wrong
# 2. Identify sources of inpurity\
# Find which entries have reco. Michels that are not matched to a true Michel\
# Visualize these entries in an event display\
# Catalogue what is going wrong
# 3. Estimate Michel energy resolution\
# Draw the reco. particle charge depositions (particle.depositions.sum()) as a function of the match true particle initial energy (truth_particle.energy_init)\
# Bin it in true energy, estimate the std in each bin
# 4. Change selection parameters
#     (muon_min_voxel_count = 30,
#     attach_threshold = 3,
#     match_threshold = 0.5)
# 5. Performance as a function of energy
# 6. Find more variables to add to [michels] table
# 7. Apply shower fudge factor on reco deposition sum
# 
# (Also do truth and reco energy spectra)

# %%



