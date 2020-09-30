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
    """Calculate Jaccard coefficient

    Ranges from 0 to 1. Higher values indicate greater similarity.

    Parameters
    ----------
    y_true : list or array
        Boolean indication of cluster membership (1 if belonging, 0 if
        not belonging) for actual labels
    y_pred : list or array
        Boolean indication of cluster membership (1 if belonging, 0 if
        not belonging) for predicted labels

    Returns
    -------
    float
        Jaccard coefficient value
    """
    if type(y_true) is list:
        y_true = np.array(y_true)

    if type(y_pred) is list:
        y_pred = np.array(y_pred)
    return float(np.sum(y_true & y_pred)) / (np.sum(y_true) + np.sum(y_pred) - np.sum(y_true & y_pred))


def all_cluster_calls(specimen_ids, morph_data, ephys_data,
                      weights=[1, 2, 5], n_cl=[10, 15, 20, 25], n_nn=[4, 7, 10]):
    """Perform several clustering algorithms and variations

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
        Combined clustering results. The index is the set of
        `specimen_ids`. Each column contains cluster labels for a
        different clustering variant.
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


def hc_nn_cluster_calls(results_df, morph_data, ephys_data,
                        n_nn=[4, 7, 10], n_cl=[10, 15, 20, 25]):
    """Add agglomerative clustering results with connectivity constraints

    Parameters
    ----------
    results_df : DataFrame
        Existing DataFrame that new clustering results are added to as
        new columns
    morph_data : array
        Specimens by morphology features array
    ephys_data : array
        Specimens by electrophysiology features array
    n_nn : list, optional
        Set of nearest-neighbor values
    n_cl : list, optional
        Set of cluster numbers

    Returns
    -------
    DataFrame
        Combined clustering results. The index is the set of
        `specimen_ids`. Each column contains cluster labels for a
        different clustering variant.
    """
    pw = metrics.pairwise.pairwise_distances(ephys_data)

    for nn in n_nn:
        knn_graph = neighbors.kneighbors_graph(pw, nn, include_self=False)
        for cl in n_cl:
            key = "hc_conn_{:d}_{:d}".format(nn, cl)
            model = cluster.AgglomerativeClustering(linkage="ward",
                                            connectivity=knn_graph,
                                            n_clusters=cl)
            model.fit(morph_data)
            results_df[key] = model.labels_

    return results_df


def hc_combo_cluster_calls(results_df, morph_data, ephys_data,
                        weights=[1, 2, 5], n_cl=[10, 15, 20, 25]):
    """Add agglomerative hierarchical clustering results

    Parameters
    ----------
    results_df : DataFrame
        Existing DataFrame that new clustering results are added to as
        new columns
    morph_data : array
        Specimens by morphology features array
    ephys_data : array
        Specimens by electrophysiology features array
    weights : list, optional
        Set of relative electrophysiology weights
    n_cl : list, optional
        Set of cluster numbers

    Returns
    -------
    DataFrame
        Combined clustering results. The index is the set of
        `specimen_ids`. Each column contains cluster labels for a
        different clustering variant.
    """
    for wt in weights:
        EM_data = np.hstack([morph_data, wt * ephys_data])
        for cl in n_cl:
            key = "hc_combo_{:g}_{:d}".format(wt, cl)
            model = cluster.AgglomerativeClustering(linkage="ward",
                                            n_clusters=cl)
            model.fit(EM_data)
            results_df[key] = model.labels_

    return results_df


def gmm_combo_cluster_calls(results_df, morph_data, ephys_data,
                        weights=[1, 2, 5], n_cl=[10, 15, 20, 25]):
    """Add Gaussian mixture model clustering results

    Parameters
    ----------
    results_df : DataFrame
        Existing DataFrame that new clustering results are added to as
        new columns
    morph_data : array
        Specimens by morphology features array
    ephys_data : array
        Specimens by electrophysiology features array
    weights : list, optional
        Set of relative electrophysiology weights
    n_cl : list, optional
        Set of cluster numbers

    Returns
    -------
    DataFrame
        Combined clustering results. The index is the set of
        `specimen_ids`. Each column contains cluster labels for a
        different clustering variant.
    """
    for wt in weights:
        EM_data = np.hstack([morph_data, wt * ephys_data])
        for cl in n_cl:
            key = "gmm_combo_{:g}_{:d}".format(wt, cl)
            model = mixture.GaussianMixture(n_components=cl, covariance_type="diag", n_init=20, random_state=0)
            results_df[key] = model.fit_predict(EM_data)

    return results_df


def spectral_combo_cluster_calls(results_df, morph_data, ephys_data,
                        weights=[1, 2, 5], n_cl=[10, 15, 20, 25]):
    """Add spectral clustering results

    Parameters
    ----------
    results_df : DataFrame
        Existing DataFrame that new clustering results are added to as
        new columns
    morph_data : array
        Specimens by morphology features array
    ephys_data : array
        Specimens by electrophysiology features array
    weights : list, optional
        Set of relative electrophysiology weights
    n_cl : list, optional
        Set of cluster numbers

    Returns
    -------
    DataFrame
        Combined clustering results. The index is the set of
        `specimen_ids`. Each column contains cluster labels for a
        different clustering variant.
    """
    for wt in weights:
        EM_data = np.hstack([morph_data, wt * ephys_data])
        for cl in n_cl:
            key = "spec_combo_{:g}_{:d}".format(wt, cl)
            model = cluster.SpectralClustering(cl, gamma=0.01, n_init=20, random_state=0)
            results_df[key] = model.fit_predict(EM_data)
    return results_df


def consensus_clusters(results, min_clust_size=3):
    """Determine consensus clusters from multiple variations

    The method iteratively divides the co-clustering matrix by Ward
    hierarchical clustering, using the co-clustering fractions as the
    distance measure. The iterative division stops when a resultant
    cluster would be smaller than `min_clust_size`. Next, co-clustering
    rates between clusters are evaluated, and clusters are merged if the
    higher of the two within-cluster rates fails to exceed the
    between-cluster rate by 25%. Sample assignments are then refined by
    reassignment to the best-matched cluster (repeated until
    convergence). This procedure is based on the one described by `Tasic
    et al. (2018) <https://www.nature.com/articles/s41586-018-0654-5>`_.

    Parameters
    ----------
    results : array
        Results of multiple clustering variants. Each column contains
        labels from a different variant.
    min_clust_size : int, optional
        Minimum size of consensus cluster

    Returns
    -------
    clust_labels : array
        Consensus cluster labels
    shared_norm : array
        Sample by sample matrix of the fraction of times samples were
        placed in the same cluster
    cc_rates : array
        Cluster by cluster matrix of the average co-clustering rates
        between cells in a pair of consensus clusters
    """
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
    """Reassign samples to the best-matched clusters

    All samples that have a better-matching cluster are reassigned at a
    time. Since reassignment changes the matching rates, the procedure
    is repeated until assignments no longer change or are identical to a
    previously encountered set of assignments (meaning that the
    algorithm has entered a cycle).

    Parameters
    ----------
    clust_labels : array
        Cluster assignments
    shared_norm : array
        Matrix of normalized cell-by-cell co-clustering rates

    Returns
    -------
    array
        New cluster assignments

    """
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
    """Calculate co-clustering rates between clusters

    Parameters
    ----------
    shared : array
        Matrix of normalized cell-by-cell co-clustering rates
    clust_labels : array
        Cluster assignments
    uniq_labels : array
        Set of unique cluster labels

    Returns
    -------
    array
        Cluster-by-cluster matrix of co-clustering rates
    """
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


def subsample_run(original_labels, specimen_ids, morph_data, ephys_data,
                  weights=[1, 2, 5], n_cl=[10, 15, 20, 25], n_nn=[4, 7, 10],
                  n_folds=10, n_iter=1, min_consensus_n=3):
    """Calculate Jaccard coefficients for subsampled clustering runs

    Parameters
    ----------
    original_labels : array
        Cluster assignments from analysis on full data set
    specimen_ids : array
        Specimen labels
    morph_data : array
        Specimen by morphology feature matrix
    ephys_data : array
        Specimen by electrophysiology feature matrix
    weights : list, optional
        Set of relative electrophysiology weights
    n_cl : list, optional
        Set of cluster numbers
    n_nn : list, optional
        Set of nearest-neighbor values
    n_folds : int, optional
        Number of subsample folds
    n_iter : int, optional
        Number of subsampled runs to perform
    min_consensus_n : int, optional
        Minimum size of consensus cluster

    Returns
    -------
    array
        Jaccard coefficients of each cluster (rows) from each run (columns).
        This array will have ``n_iter`` * ``n_folds`` columns.

    """
    run_info_list = [{
        "iter_number": i,
        "original_labels": original_labels,
        "specimen_ids": specimen_ids,
        "morph_data": morph_X,
        "ephys_data": ephys_spca,
        "weights": weights,
        "n_cl": n_cl,
        "n_nn": n_nn,
        "n_folds": n_folds,
        "min_consensus_n": min_consensus_n,
    } for i in range(n_iter)]

    p = Pool()
    logging.info("Starting multiprocessing")
    results = []
    for i, res in enumerate(p.imap_unordered(_individual_subsample_run, run_info_list, 1)):
        sys.stderr.write('\rdone {0:%}'.format(float(i + 1)/len(run_info_list)))
        results.append(res)

    jaccards = np.hstack(results)
    return jaccards


def _individual_subsample_run(run_info):
    """Perform an individual subsample run

    Used within :func:`subsample_run`
    """
    i = run_info["iter_number"]
    original_labels = run_info["original_labels"]
    specimen_ids = run_info["specimen_ids"]
    morph_data = run_info["morph_data"]
    ephys_data = run_info["ephys_data"]
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
                                              morph_data[train_index, :],
                                              ephys_data[train_index, :],
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
