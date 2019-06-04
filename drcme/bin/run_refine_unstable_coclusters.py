#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import pandas as pd
import argschema as ags
import logging


class RefineParameters(ags.ArgSchema):
    cocluster_matrix_file = ags.fields.InputFile()
    jaccards_file = ags.fields.InputFile()
    cluster_labels_file = ags.fields.InputFile()
    ordering_file = ags.fields.InputFile()

    refined_labels_file = ags.fields.OutputFile()
    refined_text_labels_file = ags.fields.OutputFile()
    refined_ordering_file = ags.fields.OutputFile()

    unstable_threshold = ags.fields.Float(default=0.5)
    coclust_threshold = ags.fields.Float(default=0.4)
    pct_needed = ags.fields.Float(default=0.33)
    me_prefix = ags.fields.String()


def intact_match_rates(clust_labels, shared, intact_clusters):
    rates = []
    for cl in intact_clusters:
        cl_mask = cl == clust_labels
        cl_cells = np.flatnonzero(cl_mask)
        for cell_index in cl_cells:
            my_mask = np.ones(shared.shape[0]).astype(bool)
            my_mask[cell_index] = False
            rates.append(np.mean(shared[cell_index, :][my_mask & cl_mask]))
    return rates


def match_rates_for_dissolved_clusters(dissolve_clusters, intact_clusters, clust_labels, shared, threshold):
    results = []
    for cl in dissolve_clusters:
        cl_cells = np.flatnonzero(cl == clust_labels)
        if len(cl_cells) == 0:
            continue
        best_match_rates = []
        for cell_index in cl_cells:
            other_rates = []
            for other_cl in intact_clusters:
                other_mask = other_cl == clust_labels
                other_rates.append(np.mean(shared[cell_index, :][other_mask]))
            best_match_rates.append(np.max(other_rates))
        results.append((cl, np.sum(np.array(best_match_rates) > threshold).astype(float) / len(best_match_rates)))
    return dict(results)


def new_labels_for_dissolved_cluster(cl, clust_labels, shared, intact_clusters):
    cl_cells = np.flatnonzero(cl == clust_labels)
    new_labelling = []
    for cell_index in cl_cells:
        other_rates = []
        for other_cl in intact_clusters:
            other_mask = other_cl == clust_labels
            other_rates.append(np.mean(shared[cell_index, :][other_mask]))
        new_cl = intact_clusters[np.argmax(other_rates)]
        new_labelling.append((cell_index, new_cl))
    new_labels = clust_labels.copy()
    for ci, new_cl in new_labelling:
        new_labels[ci] = new_cl
    return new_labels


def main(cocluster_matrix_file, jaccards_file, cluster_labels_file, ordering_file,
         refined_labels_file, refined_text_labels_file, refined_ordering_file,
         unstable_threshold, coclust_threshold, pct_needed, me_prefix,
         **kwargs):
    shared = np.loadtxt(cocluster_matrix_file)
    jaccards = np.loadtxt(jaccards_file)
    clust_labels_df = pd.read_csv(cluster_labels_file, index_col=0)
    print(clust_labels_df.head())
    clust_labels = clust_labels_df["0"].values
    new_order = np.loadtxt(ordering_file).astype(int)

    unstable_clusters = np.flatnonzero(jaccards.mean(axis=1) < unstable_threshold)
    intact_clusters = np.flatnonzero(jaccards.mean(axis=1) >= 0.5)
    logging.info("Found {:d} unstable clusters".format(len(unstable_clusters)))

    intact_rates = intact_match_rates(clust_labels, shared, intact_clusters)

    logging.info("Using coclustering threshold of {:g}".format(coclust_threshold))
    logging.info("cf. intact rates 5th percentile {:g}".format(np.percentile(intact_rates, 5)))

    keep_going = True
    refined_clust_labels = clust_labels.copy()
    while keep_going:
        mr = match_rates_for_dissolved_clusters(unstable_clusters, intact_clusters,
                                                refined_clust_labels, shared,
                                                threshold=coclust_threshold)
        if len(list(mr.keys())) == 0:
            break
        cl_for_dissolve = max(mr, key=lambda key: mr[key])
        if mr[cl_for_dissolve] < pct_needed:
            break
        logging.info("Dissolving {:d}".format(int(cl_for_dissolve)))
        refined_clust_labels = new_labels_for_dissolved_cluster(cl_for_dissolve,
                                                                refined_clust_labels,
                                                                shared, intact_clusters)

    refined_order = np.lexsort((clust_labels, refined_clust_labels))
    refined_relabel = {v: i + 1 for i, v in enumerate(np.sort(np.unique(refined_clust_labels)))}
    refined_text_labels = ["{:s}_{:d}".format(me_prefix, refined_relabel[v]) for v in refined_clust_labels]

    pd.DataFrame(refined_clust_labels, index=clust_labels_df.index.values).to_csv(refined_labels_file)
    pd.DataFrame(refined_text_labels, index=clust_labels_df.index.values).to_csv(refined_text_labels_file)
    np.savetxt(refined_ordering_file, refined_order, fmt="%d")


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=RefineParameters)
    main(**module.args)
