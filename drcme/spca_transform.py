#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import pandas as pd


def orig_mean_and_std_for_zscore(spca_results, orig_data, spca_zht_params, pev_threshold=0.01):
    Z_list = []
    for ds in orig_data:
        data = ds["data"]
        for k in ds["part_keys"]:
            _, _, _, indices = spca_zht_params[k]
            d = data[:, indices]
            above_thresh = spca_results[k]["pev"] >= pev_threshold
            Z = d.dot(spca_results[k]["loadings"][:, above_thresh])
            if np.any(np.isnan(Z)):
                print("NaNs found", k)
            Z_list.append(Z)

    combo_orig = np.hstack(Z_list)
    return combo_orig.mean(axis=0), combo_orig.std(axis=0)


def spca_transform_new_data(spca_results, new_data, spca_zht_params, orig_mean, orig_std, pev_threshold=0.01):
    """Transform and z-score new data by previously-determined components"""
    Z_list = []
    for ds in new_data:
        data = ds["data"]
        for k in ds["part_keys"]:
            _, _, _, indices = spca_zht_params[k]
            d = data[:, indices]
            above_thresh = spca_results[k]["pev"] >= pev_threshold
            Z = d.dot(spca_results[k]["loadings"][:, above_thresh])
            if np.any(np.isnan(Z)):
                print("NaNs found", k)
            Z_list.append(Z)

    combo_new = np.hstack(Z_list)
    combo = (combo_new - orig_mean) / orig_std
    return combo
