#!/usr/bin/env python

from __future__ import print_function
from builtins import zip
from builtins import range
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import scipy
import os.path
import joblib
from scipy.cluster import hierarchy
import argschema as ags
import logging

class MergeParameters(ags.ArgSchema):
    project = ags.fields.String(default="T301")
    gmm_types = ags.fields.List(ags.fields.String, default=["diag", "diag", "diag"])


def log_p(x):
    return np.log(np.maximum(x, 1e-300))


def entropy_merges(results_dir, project="T301", gmm_type="diag", piecewise_components=2):
    data = pd.read_csv(os.path.join("..", "dev", results_dir, "sparse_pca_components_{:s}.csv".format(project)), index_col=0)
    gmm = joblib.load(os.path.join("..", "dev", results_dir, "best_gmm_{:s}.pkl".format(project)))
    my_gmm = gmm[gmm_type]
    combo = data.values
    tau = my_gmm.predict_proba(combo)
    labels = my_gmm.predict(combo)
    K_bic = max(labels) + 1

    return entropy_combi(tau_labels, K_bic, piecewise_components=piecewise_components)


def entropy_specific_merges(tau, labels, K_bic, clusters_to_merge):
    entropy = -np.sum(np.multiply(tau, log_p(tau)))
    prior_merges = []
    merges_by_names = []
    entropies = [entropy]
    Ks = [K_bic]
    n_to_merge = len(clusters_to_merge)
    n_merge_tracker = [0]
    orig_col_names = np.arange(K_bic)

    for K in range(K_bic, K_bic - n_to_merge, -1):
        merge_matrix = np.identity(K_bic, dtype=int)
        merge_col_names = orig_col_names.copy()
        for merger in prior_merges:
            i, j = merger
            merge_matrix[:, i] = merge_matrix[:, i] | merge_matrix[:, j]
            mask = np.ones(merge_matrix.shape[1], dtype=bool)
            mask[j] = False
            merge_matrix = merge_matrix[:, mask]
            merge_col_names = merge_col_names[mask]
        labels = np.argmax(np.dot(tau, merge_matrix), axis=1)

        ent_current = np.inf
        candidate_cols = np.arange(K)[np.isin(merge_col_names, clusters_to_merge)]
        logging.debug("candidates left")
        logging.debug(merge_col_names[np.isin(merge_col_names, clusters_to_merge)])
        remaining_cols = np.arange(K)[~np.isin(merge_col_names, clusters_to_merge)]
        for i in remaining_cols:
            for j in candidate_cols:
                new_merge_matrix = merge_matrix.copy()
                new_merge_matrix[:, i] = merge_matrix[:, i] | merge_matrix[:, j]
                mask = np.ones(K, dtype=bool)
                mask[j] = False
                new_merge_matrix = new_merge_matrix[:, mask]
                tau_m = np.dot(tau, new_merge_matrix)
                ent = -np.sum(np.multiply(tau_m, log_p(tau_m)))
                if ent < ent_current:
                    ent_current = ent
                    merger = (i, j)
                    n_merged = np.sum(labels == i) + np.sum(labels == j)
        prior_merges.append(merger)
        merges_by_names.append([merge_col_names[i].item() for i in merger])
        entropies.append(ent_current)
        Ks.append(K-1)
        n_merge_tracker.append(n_merged)

    merge_info = {
        "entropies": entropies,
        "merge_sequence": prior_merges,
        "merges_by_names": merges_by_names,
        "K_bic": K_bic,
        "Ks": Ks,
        "cumul_merges": np.cumsum(n_merge_tracker)
    }

    merge_matrix = np.identity(K_bic, dtype=int)
    for merger in prior_merges:
        i, j = merger
        merge_matrix[:, i] = merge_matrix[:, i] | merge_matrix[:, j]
        mask = np.ones(merge_matrix.shape[1], dtype=bool)
        mask[j] = False
        merge_matrix = merge_matrix[:, mask]

    tau_merged = np.dot(tau, merge_matrix)
    new_labels = np.argmax(tau_merged, axis=1)

    return merge_info, new_labels, tau_merged, merge_matrix


def entropy_combi(tau, labels, K_bic, piecewise_components=2):
    entropy = -np.sum(np.multiply(tau, log_p(tau)))
    prior_merges = []
    entropies = [entropy]
    Ks = [K_bic]
    n_merge_tracker = [0]
    if K_bic <= 3:
        print("Too few clusters to assess merging")
        merge_info = {
            "entropies": entropies,
            "cumul_merges": None,
            "best_fits": None,
#             "fit2": None,
            "cp": None,
            "merge_sequence": None,
            "K_bic": K_bic,
        }
        return merge_info, labels, tau, None

    for K in range(K_bic, 1, -1):
        merge_matrix = np.identity(K_bic, dtype=int)
        for merger in prior_merges:
            i, j = merger
            merge_matrix[:, i] = merge_matrix[:, i] | merge_matrix[:, j]
            mask = np.ones(merge_matrix.shape[1], dtype=bool)
            mask[j] = False
            merge_matrix = merge_matrix[:, mask]
        labels = np.argmax(np.dot(tau, merge_matrix), axis=1)

        ent_current = np.inf
        for i in range(K-1):
            for j in range(i + 1, K):
                new_merge_matrix = merge_matrix.copy()
                new_merge_matrix[:, i] = merge_matrix[:, i] | merge_matrix[:, j]
                mask = np.ones(K, dtype=bool)
                mask[j] = False
                new_merge_matrix = new_merge_matrix[:, mask]
                tau_m = np.dot(tau, new_merge_matrix)
                ent = -np.sum(np.multiply(tau_m, log_p(tau_m)))
                if ent < ent_current:
                    ent_current = ent
                    merger = (i, j)
                    n_merged = np.sum(labels == i) + np.sum(labels == j)
        prior_merges.append(merger)

        entropies.append(ent_current)
        Ks.append(K-1)
        n_merge_tracker.append(n_merged)

    cumul_merges = np.cumsum(n_merge_tracker)
    best_fits, cp = fit_piecewise(cumul_merges, entropies, piecewise_components)
    merge_info = {
        "entropies": entropies,
        "cumul_merges": cumul_merges,
        "best_fits": best_fits,
#         "fit2": fit2,
        "cp": cp,
        "merge_sequence": prior_merges[:cp[0]],
        "K_bic": K_bic,
    }

    merge_matrix = np.identity(K_bic, dtype=int)
    for merger in prior_merges[:cp[0]]:
        i, j = merger
        merge_matrix[:, i] = merge_matrix[:, i] | merge_matrix[:, j]
        mask = np.ones(merge_matrix.shape[1], dtype=bool)
        mask[j] = False
        merge_matrix = merge_matrix[:, mask]

    tau_merged = np.dot(tau, merge_matrix)
    new_labels = np.argmax(tau_merged, axis=1)

    return merge_info, new_labels, tau_merged, merge_matrix


def fit_piecewise(cumul_merges, entropies, n_parts):
    total_err = np.inf

    if len(entropies) < n_parts + 2:
        logging.info("Not enough clusters for piecewise fit")
        return (None,), (0,)

    if n_parts == 2:
        for c in range(1, len(entropies)-1):
            x1 = cumul_merges[:c + 1]
            x2 = cumul_merges[c:]
            y1 = entropies[:c + 1]
            y2 = entropies[c:]

            A1 = np.vstack([x1, np.ones(len(x1))]).T
            A2 = np.vstack([x2, np.ones(len(x2))]).T

            fit1 = np.linalg.lstsq(A1, y1, rcond=-1)
            fit2 = np.linalg.lstsq(A2, y2, rcond=-1)

            err = fit1[1] + fit2[1]

            if err < total_err:
                total_err = err
                best_fits = (fit1[0], fit2[0])
                cp = (c,)
    elif n_parts == 3:
        for c in range(1, len(entropies) - 3):
            for d in range(c + 1, len(entropies)-1):
                x1 = cumul_merges[:c + 1]
                x2 = cumul_merges[c:d + 1]
                x3 = cumul_merges[d:]
                y1 = entropies[:c + 1]
                y2 = entropies[c:d + 1]
                y3 = entropies[d:]

                A1 = np.vstack([x1, np.ones(len(x1))]).T
                A2 = np.vstack([x2, np.ones(len(x2))]).T
                A3 = np.vstack([x3, np.ones(len(x3))]).T

                fit1 = np.linalg.lstsq(A1, y1, rcond=-1)
                fit2 = np.linalg.lstsq(A2, y2, rcond=-1)
                fit3 = np.linalg.lstsq(A3, y3, rcond=-1)

                err = fit1[1] + fit2[1] + fit3[1]

                if err < total_err:
                    total_err = err
                    best_fits = (fit1[0], fit2[0], fit3[0])
                    cp = (c, d)
    else:
        raise("Wrong value for n_parts")

    return best_fits, cp


def order_new_labels(new_labels, tau_merged, data):
    uniq_labels = np.unique(new_labels)
    uniq_labels = uniq_labels[~np.isnan(uniq_labels)]
    n_cl = len(uniq_labels)
    centroids = np.zeros((n_cl, data.shape[1]))
    for i in range(n_cl):
        centroids[i, :] = np.mean(data.values[new_labels == i, :], axis=0)
    Z = hierarchy.linkage(centroids, method="ward")
    D = hierarchy.dendrogram(Z, no_plot=True)
    leaves = np.array(D["leaves"])
    # put singletons at the end
#     is_singleton = np.array([np.sum(new_labels == l) == 1 for l in leaves])
#     leaves = np.hstack([leaves[~is_singleton], leaves[is_singleton]])
    new_labels_reorder_dict = {d: i for i, d in enumerate(leaves)}
    tau_merged = tau_merged[:, leaves]
    new_labels_reorder = [new_labels_reorder_dict[d] if not np.isnan(d) else d for d in new_labels]

    return new_labels_reorder, tau_merged, new_labels_reorder_dict, leaves


def main():
    module = ags.ArgSchemaParser(schema_type=MergeParameters)
    project = module.args["project"]
    gmm_types = module.args["gmm_types"]

    sub_dirs = [s.format(project) for s in ["all_{:s}", "exc_{:s}", "inh_{:s}"]]
    piecewise_components = [2, 2, 3]
    for sub_dir, gmm_type, pw_comp in zip(sub_dirs, gmm_types, piecewise_components):
        print("merging for ", sub_dir, "with", gmm_type)
        merge_info, new_labels, tau_merged, _ = entropy_merges(sub_dir, project, gmm_type=gmm_type, piecewise_components=pw_comp)
        print(merge_info)
        data = pd.read_csv(os.path.join(sub_dir, "sparse_pca_components_{:s}.csv".format(project)), index_col=0)
        new_labels, tau_merged, _, _ = order_new_labels(new_labels, tau_merged, data)

        np.savetxt(os.path.join(sub_dir, "post_merge_proba.txt"), tau_merged)
        np.save(os.path.join(sub_dir, "post_merge_cluster_labels.npy"), new_labels)
        df = pd.read_csv(os.path.join(sub_dir, "all_tsne_coords_{:s}.csv".format(project)))
        df["clustering_3"] = new_labels
        df.to_csv(os.path.join(sub_dir, "all_tsne_coords_{:s}_plus.csv".format(project)))


if __name__ == "__main__":
    main()