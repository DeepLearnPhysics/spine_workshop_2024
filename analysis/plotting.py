import re, os, sys
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix


def make_table(df_numu, true_mask, pred_mask):

    truth_labels = {}
    reco_labels = {}

    df_analysis = df_numu[['Index', 'file_index', 'true_interaction_id', 'reco_interaction_id']]
    df_analysis['Truth'] = -1
    df_analysis['Prediction'] = -1
    truth_labels[-1] = 'Other'

    # True Cosmics
    mask = df_numu['true_nu_id'] < 0
    df_analysis.loc[mask, 'Truth'] = 0
    truth_labels[0] = 'Cosmics'

    # True Signal
    # true_mask = manager.true_def_t2r[-1].astype(bool)
    df_analysis.loc[true_mask, 'Truth'] = 1
    truth_labels[1] = '$\\nu_e$ 1eNp'

    # True Other Nue
    nue_mask = ((df_numu['true_nu_pdg_code'] == 12) | (df_numu['true_nu_pdg_code'] == -12)).astype(bool)
    mask = nue_mask & (df_numu['true_nu_current_type'] == 0)
    mask = mask & ~true_mask
    df_analysis.loc[mask, 'Truth'] = 2
    truth_labels[2] = '$\\nu_e$ CC'

    # True Other Nue
    nue_mask = ((df_numu['true_nu_pdg_code'] == 12) | (df_numu['true_nu_pdg_code'] == -12)).astype(bool)
    mask = nue_mask & (df_numu['true_nu_current_type'] == 1)
    mask = mask & ~true_mask
    df_analysis.loc[mask, 'Truth'] = 3
    truth_labels[3] = '$\\nu_e$ NC'

    # True Numu CC
    numu_mask = ((df_numu['true_nu_pdg_code'] == 14) | (df_numu['true_nu_pdg_code'] == -14)).astype(bool)
    mask = numu_mask & (df_numu['true_nu_current_type'] == 0)
    mask = mask & ~true_mask
    df_analysis.loc[mask, 'Truth'] = 4
    truth_labels[4] = '$\\nu_\\mu$ CC'

    # True Numu NC
    numu_mask = ((df_numu['true_nu_pdg_code'] == 14) | (df_numu['true_nu_pdg_code'] == -14)).astype(bool)
    mask = numu_mask & (df_numu['true_nu_current_type'] == 1)
    mask = mask & ~true_mask
    df_analysis.loc[mask, 'Truth'] = 5
    truth_labels[5] = '$\\nu_\\mu$ NC'

    # pred_mask = manager.reco_def_t2r[-1].astype(bool)
    df_analysis.loc[pred_mask, 'Prediction'] = 1
    reco_labels[1] = 'Signal'

    # pred_mask = manager.reco_def_t2r[-1].astype(bool)
    df_analysis.loc[~pred_mask, 'Prediction'] = 0
    reco_labels[0] = 'Background'
    
    return df_analysis, truth_labels, reco_labels