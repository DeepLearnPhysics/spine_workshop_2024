import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import tenumerate

from collections import namedtuple

class Cut:
    
    def __call__(self):
        raise NotImplementedError()
    
    
CutResult = namedtuple('Cut', ['name', 'efficiency', 'num_efficiency', 'purity', 'num_purity'])
    
    
class SelectionManager:
    
    def __init__(self, df_intrs_t2r, df_intrs_r2t):
        self.df_intrs_t2r = df_intrs_t2r
        self.df_intrs_r2t = df_intrs_r2t
        
        self.length_t2r = self.df_intrs_t2r.shape[0]
        self.length_r2t = self.df_intrs_r2t.shape[0]

        self.df_filters_true_t2r = {}
        self.df_filters_reco_t2r = {}
        
        self.df_filters_true_r2t = {}
        self.df_filters_reco_r2t = {}
        
        self.filters = []
        
    def register_filter(self, f):
        self.filters.append(f)
        
        self.df_filters_true_t2r[f.name] = np.zeros(self.length_t2r, dtype=bool)
        self.df_filters_true_r2t[f.name] = np.zeros(self.length_r2t, dtype=bool)
        
        self.df_filters_reco_t2r[f.name] = np.zeros(self.length_t2r, dtype=bool)
        self.df_filters_reco_r2t[f.name] = np.zeros(self.length_r2t, dtype=bool)
        
    def _process_t2r(self):
        
        for i, ia in tenumerate(self.df_intrs_t2r.itertuples(index=False, name='Interaction'), total=self.length_t2r):
            # index = (ia.Index, ia.file_index, ia.true_interaction_id)
            for f in self.filters:
                self.df_filters_true_t2r[f.name][i] = f(ia, mode='true')
                self.df_filters_reco_t2r[f.name][i] = f(ia, mode='reco')
                
    def _process_r2t(self):
        
        for i, ia in tenumerate(self.df_intrs_r2t.itertuples(index=False, name='Interaction'), total=self.length_r2t):
            # index = (ia.Index, ia.file_index, ia.reco_interaction_id)
            for f in self.filters:
                self.df_filters_true_r2t[f.name][i] = f(ia, mode='true')
                self.df_filters_reco_r2t[f.name][i] = f(ia, mode='reco')
                
    def process(self):
        
        self.num_filters = len(self.filters)
        
        self._process_r2t()
        self._process_t2r()
        
    def evaluate(self):
        
        self.true_def_t2r = np.zeros((len(self.df_filters_true_t2r), self.length_t2r), dtype=bool)
        self.true_def_r2t = np.zeros((len(self.df_filters_true_r2t), self.length_r2t), dtype=bool)
        
        self.reco_def_t2r = np.zeros((len(self.df_filters_reco_t2r), self.length_t2r), dtype=bool)
        self.reco_def_r2t = np.zeros((len(self.df_filters_reco_r2t), self.length_r2t), dtype=bool)
        
        assert self.num_filters == len(self.true_def_t2r)
        assert self.num_filters == len(self.true_def_r2t)
        assert self.num_filters == len(self.reco_def_t2r)
        assert self.num_filters == len(self.reco_def_r2t)
        
        for i, f in enumerate(self.filters):
            self.true_def_t2r[i] = self.df_filters_true_t2r[f.name]
            self.true_def_r2t[i] = self.df_filters_true_r2t[f.name]
            
        for i, f in enumerate(self.filters):
            self.reco_def_t2r[i] = self.df_filters_reco_t2r[f.name]
            self.reco_def_r2t[i] = self.df_filters_reco_r2t[f.name]
            
        self.true_def_t2r = np.cumprod(self.true_def_t2r, axis=0)
        self.true_def_r2t = np.cumprod(self.true_def_r2t, axis=0)
        self.reco_def_t2r = np.cumprod(self.reco_def_t2r, axis=0)
        self.reco_def_r2t = np.cumprod(self.reco_def_r2t, axis=0)
        
        out = []
        
        # First define true interaction that are reconstructible
        truth_t2r = self.true_def_t2r[-1]
        truth_r2t = self.true_def_r2t[-1]
        
        
        for i, f in enumerate(self.filters):
            eff_num, eff_den = np.sum(self.reco_def_t2r[i] * truth_t2r), np.sum(truth_t2r)
            eff = eff_num / eff_den if eff_den > 0 else 0

            pur_num, pur_den = np.sum(self.reco_def_r2t[i] * truth_r2t), np.sum(self.reco_def_r2t[i])
            pur = pur_num / pur_den if pur_den > 0 else 0

            out.append(CutResult(f.name, eff, f"{eff_num} / {eff_den}", pur, f"{pur_num} / {pur_den}"))
            
        return out
    
    def get_selection(self, idx):
        
        true_t2r_bool = self.true_def_t2r[idx].astype(bool)
        true_r2t_bool = self.true_def_r2t[idx].astype(bool)
        reco_t2r_bool = self.reco_def_t2r[idx].astype(bool)
        reco_r2t_bool = self.reco_def_r2t[idx].astype(bool)
        
        t2r_tp_mask = np.logical_and(true_t2r_bool, reco_t2r_bool)
        r2t_tp_mask = np.logical_and(true_r2t_bool, reco_r2t_bool)
        
        t2r_fn_mask = np.logical_and(true_t2r_bool, ~reco_t2r_bool)
        r2t_fn_mask = np.logical_and(true_r2t_bool, ~reco_r2t_bool)
        
        t2r_fp_mask = np.logical_and(~true_t2r_bool, reco_t2r_bool)
        r2t_fp_mask = np.logical_and(~true_r2t_bool, reco_r2t_bool)
        
        out = {}
        
        out['t2r_tp'] = self.df_intrs_t2r.loc[t2r_tp_mask]
        out['r2t_tp'] = self.df_intrs_r2t.loc[r2t_tp_mask]
        out['t2r_fn'] = self.df_intrs_t2r.loc[t2r_fn_mask]
        out['r2t_fn'] = self.df_intrs_r2t.loc[r2t_fn_mask]
        out['t2r_fp'] = self.df_intrs_t2r.loc[t2r_fp_mask]
        out['r2t_fp'] = self.df_intrs_r2t.loc[r2t_fp_mask]
        
        return out
        
        
class TrueSignalCut(Cut):
    
    name = 'true_signal_cut'
    _use_particles = False
    
    def __init__(self, pdg_code=12):
        super(TrueSignalCut, self).__init__()
        self.pdg_code = pdg_code
        
    def __call__(self, ia):
        out = ia.true_nu_id > -1 and ia.true_nu_pdg_code == self.pdg_code
        out = out and (ia.true_num_primary_electrons == 1) and \
                        (ia.true_num_primary_protons >= 1) and \
                        (ia.true_num_primary_photons == 0) and \
                        (ia.true_num_primary_muons == 0) and \
                        (ia.true_num_primary_pions == 0)
        return out
    
    
class FinalStateTopologyCut(Cut):
    
    name = 'final_state_topology_cut'
    _use_particles = False
    
    def __init__(self):
        super(FinalStateTopologyCut, self).__init__()
            
    def __call__(self, ia, mode):
        if mode == 'true':
            out = (ia.true_num_primary_electrons == 1) and \
                (ia.true_num_primary_protons >= 1) and \
                (ia.true_num_primary_photons == 0) and \
                (ia.true_num_primary_muons == 0) and \
                (ia.true_num_primary_pions == 0)
        elif mode == 'reco':
            out = (ia.reco_num_primary_electrons == 1) and \
                (ia.reco_num_primary_protons >= 1) and \
                (ia.reco_num_primary_photons == 0) and \
                (ia.reco_num_primary_muons == 0) and \
                (ia.reco_num_primary_pions == 0)
        else:
            raise ValueError('Invalid mode')
        return out
        

class ContainmentCut(Cut):
    
    name = 'containment_cut'
    _use_particles = False
    
    def __init__(self):
        super(ContainmentCut, self).__init__()
        
    def __call__(self, ia, mode):
        if mode == 'true':
            out = ia.true_interaction_is_contained
        elif mode == 'reco':
            out = ia.reco_interaction_is_contained
        else:
            raise ValueError('Invalid mode')
        return out
    
class FiducialCut(Cut):
        
    name = 'fiducial_cut'
    _use_particles = False
    
    def __init__(self):
        super(FiducialCut, self).__init__()
        
    def __call__(self, ia, mode):
        if mode == 'true':
            out = ia.true_interaction_is_fiducial
        elif mode == 'reco':
            out = ia.reco_interaction_is_fiducial
        else:
            raise ValueError('Invalid mode')
        return out
    
    
class TruthNeutrinoCut(Cut):
    
    name = 'truth_neutrino_cut'
    _use_particles = False
    
    def __init__(self, pdg_code=12):
        super(TruthNeutrinoCut, self).__init__()
        self.pdg_code = pdg_code
        
    def __call__(self, ia, mode):
        if mode == 'true':
            out = ia.true_nu_id > -1 and (ia.true_nu_pdg_code == self.pdg_code or ia.true_nu_pdg_code == -self.pdg_code)
        elif mode == 'reco':
            out = True
        else:
            raise ValueError('Invalid mode')
        return out
    
    
class RecoFlashCut(Cut):
        
    name = 'reco_flash_cut'
    _use_particles = False
    
    def __init__(self, window_lb=0.0, window_ub=1.6):
        super(RecoFlashCut, self).__init__()
        self.window_lb = window_lb
        self.window_ub = window_ub
        
    def __call__(self, ia, mode):
        if mode == 'true':
            out = True
        elif mode == 'reco':
            out = ia.reco_flash_time >= self.window_lb and ia.reco_flash_time <= self.window_ub
        else:
            raise ValueError('Invalid mode')
        return out