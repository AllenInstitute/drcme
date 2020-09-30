"""
"""

import numpy as np
import pandas as pd

import sklearn.metrics as metrics
import sklearn.cluster as cluster
import sklearn.neighbors as neighbors
import sklearn.manifold as manifold
import sklearn.mixture as mixture
import sklearn.model_selection as model_selection

import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as distance

from hashlib import sha1
from multiprocessing import Pool
import logging
import sys


def clustjaccard(y_true, y_pred):
    if type(y_true) is list:
        y_true = np.array(y_true)

    if type(y_pred) is list:
        y_pred = np.array(y_pred)
    return float(np.sum(y_true & y_pred)) / (np.sum(y_true) + np.sum(y_pred) - np.sum(y_true & y_pred))


def all_cluster_calls(specimen_ids, morph_data, ephys_data,
                      weights=[1, 2, 5], n_cl=[10, 15, 20, 25], n_nn=[4, 7, 10]):
    """ Perform several clustering algorithms and variations

    Parameters
    ----------
    specimen_ids : array
        Specimen labels of `morph_data` and `ephys_data`
    morph_data : array
        Specimens by morphology features array
    ephys_data : array
        Specimens by electrophysiology features array
    weights : list, optional
        Set of relative electrophysiology weights
    n_cl : list, optional
        Set of cluster numbers
    n_nn : list, optional
        Set of nearest-neighbor values

    Returns
    -------
    DataFrame
        Combined clustering results. The index is the set of `specimen_ids`.
    """
    results_df = pd.DataFrame({"specimen_id": specimen_ids}).set_index("specimen_id")

    results_df = hc_nn_cluster_calls(results_df, morph_data, ephys_data,
                                     n_nn=n_nn, n_cl=n_cl)
    results_df = hc_combo_cluster_calls(results_df, morph_data, ephys_data,
                                        weights=weights, n_cl=n_cl)
    results_df = gmm_combo_cluster_calls(results_df, morph_data, ephys_data,
                                         weights=weights, n_cl=n_cl)
    results_df = spectral_combo_cluster_calls(results_df, morph_data, ephys_data,
                                              weights=weights, n_cl=n_cl)

    return results_df


def hc_nn_cluster_calls(results_df, morph_X, ephys_spca,
                        n_nn=[4, 7, 10], n_cl=[10, 15, 20, 25]):
    pw = metrics.pairwise.pairwise_distances(ephys_spca)

    for nn in n_nn:
        knn_graph = neighbors.kneighbors_graph(pw, nn, include_self=False)
        for cl in n_cl:
            key = "hc_conn_{:d}_{:d}".format(nn, cl)
            model = cluster.AgglomerativeClustering(linkage="ward",
                                            connectivity=knn_graph,
                                            n_clusters=cl)
            model.fit(morph_X)
            results_df[key] = model.labels_

    return results_df


def hc_combo_cluster_calls(results_df, morph_X, ephys_spca,
                        weights=[1, 2, 5], n_cl=[10, 15, 20, 25]):
    for wt in weights:
        EM_data = np.hstack([morph_X, wt * ephys_spca])
        for cl in n_cl:
            key = "hc_combo_{:g}_{:d}".format(wt, cl)
            model = cluster.AgglomerativeClustering(linkage="ward",
                                            n_clusters=cl)
            model.fit(EM_data)
            results_df[key] = model.labels_

    return results_df


def gmm_combo_cluster_calls(results_df, morph_X, ephys_spca,
                        weights=[1, 2, 5], n_cl=[10, 15, 20, 25]):
    for wt in weights:
        EM_data = np.hstack([morph_X, wt * ephys_spca])
        for cl in n_cl:
            key = "gmm_combo_{:g}_{:d}".format(wt, cl)
            model = mixture.GaussianMixture(n_components=cl, covariance_type="diag", n_init=20, random_state=0)
            results_df[key] = model.fit_predict(EM_data)

    return results_df


def spectral_combo_cluster_calls(results_df, morph_X, ephys_spca,
                        weights=[1, 2, 5], n_cl=[10, 15, 20, 25]):
    for wt in weights:
        EM_data = np.hstack([morph_X, wt * ephys_spca])
        for cl in n_cl:
            key = "spec_combo_{:g}_{:d}".format(wt, cl)
            model = cluster.SpectralClustering(cl, gamma=0.01, n_init=20, random_state=0)
            results_df[key] = model.fit_predict(EM_data)
    return results_df


def consensus_clusters(results, min_clust_size = 3):
    n_cells = results.shape[0]
    shared = np.zeros((n_cells, n_cells))
    for i in range(shared.shape[0]):
        for j in range(i, shared.shape[0]):
            shared_count = np.sum(results[i, :] == results[j, :])
            shared[i, j] = shared_count
            shared[j, i] = shared_count

    shared_norm = shared / shared.max()

    clust_labels = np.zeros((shared.shape[0]))
    keep_going = True

    while keep_going:
        uniq_labels = np.unique(clust_labels)
        new_labels = np.zeros_like(clust_labels)
        for l in uniq_labels:
            cl_mask = clust_labels == l
            X = shared_norm[cl_mask, :][:, cl_mask]
            Z = hierarchy.linkage(X, method="ward")
            sub_labels = hierarchy.fcluster(Z, t=2, criterion="maxclust")
            if (np.sum(sub_labels == 1) < min_clust_size) or (np.sum(sub_labels == 2) < min_clust_size):
                # Don't split if it produces clusters that are too small
                sub_labels = 2 * clust_labels[cl_mask]
            else:
                sub_labels += (2 * int(l)) - 1

            new_labels[cl_mask] = sub_labels

        if metrics.adjusted_rand_score(clust_labels, new_labels) == 1:
            keep_going = False
        clust_labels = new_labels

    logging.debug(f"{len(np.unique(clust_labels))} after iterative splitting")

    keep_going = True
    while keep_going:
        # Check within and against
        uniq_labels = np.unique(clust_labels)
        logging.debug(f"examining merges for {len(uniq_labels)} labels")
        cc_rates = coclust_rates(shared_norm, clust_labels, uniq_labels)
        merges = []
        for i, l in enumerate(uniq_labels):
            for j, m in enumerate(uniq_labels[i + 1:]):
                Nll = cc_rates[i, i]
                Nmm = cc_rates[i + j + 1, i + j + 1]
                Nlm = cc_rates[i, i + j + 1]
                if Nlm > np.max([Nll, Nmm]) - 0.25:
                    merges.append((l, m, Nlm))

        if len(merges) == 0:
            keep_going = False
        else:
            best_cross = 0.
            for l, m, nlm in merges:
                if nlm > best_cross:
                    best_cross = nlm
                    merge = (l, m)
            l, m = merge
            clust_labels[clust_labels == m] = l

    clust_labels = refine_assignments(clust_labels, shared_norm)
    # Clean up the labels
    new_map = {v: i for i, v in enumerate(np.sort(np.unique(clust_labels)))}
    clust_labels = np.array([new_map[v] for v in clust_labels])
    uniq_labels = np.unique(clust_labels)
    cc_rates = coclust_rates(shared_norm, clust_labels, uniq_labels)

    return clust_labels, shared_norm, cc_rates


def refine_assignments(clust_labels, shared_norm):
    # Refine individual cell assignments
    keep_going = True
    uniq_labels = np.sort(np.unique(clust_labels))
    reassignments = []
    state_hashes = []
    while keep_going:
        last_reassignments = reassignments
        reassignments = []
        # Check within and against
        cl_masks = np.zeros((shared_norm.shape[0], len(uniq_labels)))
        for i, l in enumerate(uniq_labels):
            cl_masks[:, i] = (clust_labels == l).astype(int)
        cl_n = cl_masks.sum(axis=0)

        cl_sums = shared_norm @ cl_masks
        self_vals = np.diag(shared_norm)[:, np.newaxis] * cl_masks
        self_masks = (self_vals > 0).astype(float)
        cl_adj_sums = cl_sums - self_vals
        cl_adj_n = cl_n[np.newaxis, :] - self_masks
        cl_adj_n[cl_adj_n == 0] = 1 # avoid divide by zero

        cl_rates = cl_adj_sums / cl_adj_n
        rate_argmax = np.argmax(cl_rates, axis=1)
        assign_argmax = np.argmax(self_masks, axis=1)
        switch_candidates = np.flatnonzero(rate_argmax != assign_argmax)
        if len(switch_candidates) == 0:
            keep_going = False
        else:
            rate_max = np.max(cl_rates, axis=1)
            switch_ind = switch_candidates[np.argmax(rate_max[switch_candidates])]
            new_assignment = uniq_labels[rate_argmax[switch_ind]]
            logging.debug(f"switching {switch_ind} to {new_assignment}")
            clust_labels[switch_ind] = new_assignment

        state_hash = sha1(clust_labels).hexdigest()
        if state_hash in state_hashes: # have we encountered this set of assignments before?
            keep_going = False
        else:
            state_hashes.append(state_hash)

    return clust_labels


def coclust_rates(shared, clust_labels, uniq_labels):
        cc_rates = np.zeros((len(uniq_labels), len(uniq_labels)))
        for i, l in enumerate(uniq_labels):
            mask_l = clust_labels == l
            for j, m in enumerate(uniq_labels[i:]):
                mask_m = clust_labels == m
                X = shared[mask_l, :][:, mask_m]
                if l == m:
                    ind1, ind2 = np.tril_indices(X.shape[0], k=-1)
                    X = X[ind1, :][:, ind2]
                if X.size > 0:
                    cc_rates[i, i + j] = cc_rates[i + j, i] = X.mean()
                else:
                    cc_rates[i, i + j] = cc_rates[i + j, i] = 0

        return cc_rates


def subsample_run(original_labels, specimen_ids, morph_X, ephys_spca,
                  weights=[1, 2, 5], n_cl=[10, 15, 20, 25], n_nn=[4, 7, 10],
                  n_folds=10, n_iter=1, min_consensus_n=3):

    run_info_list = [{
        "iter_number": i,
        "original_labels": original_labels,
        "specimen_ids": specimen_ids,
        "morph_X": morph_X,
        "ephys_spca": ephys_spca,
        "weights": weights,
        "n_cl": n_cl,
        "n_nn": n_nn,
        "n_folds": n_folds,
        "min_consensus_n": min_consensus_n,
    } for i in range(n_iter)]


#     results = map(individual_subsample_run, run_info_list)
    p = Pool()
    logging.info("Starting multiprocessing")
    results = []
    for i, res in enumerate(p.imap_unordered(individual_subsample_run, run_info_list, 1)):
        sys.stderr.write('\rdone {0:%}'.format(float(i + 1)/len(run_info_list)))
        results.append(res)

    jaccards = np.hstack(results)
    return jaccards


def individual_subsample_run(run_info):
    i = run_info["iter_number"]
    original_labels = run_info["original_labels"]
    specimen_ids = run_info["specimen_ids"]
    morph_X = run_info["morph_X"]
    ephys_spca = run_info["ephys_spca"]
    weights = run_info["weights"]
    n_cl = run_info["n_cl"]
    n_nn = run_info["n_nn"]
    n_folds = run_info["n_folds"]
    min_consensus_n = run_info["min_consensus_n"]

    orig_labels_uniq = np.sort(np.unique(original_labels))

    jaccards = np.zeros((len(orig_labels_uniq), n_folds))

    kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=i)
    counter = 0
    for train_index, _ in kf.split(original_labels):
        logging.info("starting {:d} {:d}".format(i, counter))
        subsample_results = all_cluster_calls(specimen_ids[train_index],
                                              morph_X[train_index, :],
                                              ephys_spca[train_index, :],
                                              weights=weights,
                                              n_cl=n_cl,
                                              n_nn=n_nn)
        subsample_labels, _, _ = consensus_clusters(
            subsample_results.values[:, 1:], min_clust_size=min_consensus_n)

        sub_uniq = np.sort(np.unique(subsample_labels))
        for ii, orig_cl in enumerate(orig_labels_uniq):
            jacc = []
            y_orig = original_labels[train_index] == orig_cl
            for sub_cl in sub_uniq:
                y_sub = subsample_labels == sub_cl
                jacc.append(clustjaccard(y_orig, y_sub))
            jaccards[ii, counter] = np.max(jacc)
        counter += 1

    return jaccards
