"""
Script to merge cells from unstable clusters into the most similar stable ones.

The script identifies unstable clusters by the Jaccard index falling below a specified
threshold (``unstable_threshold``). It then sees how similar the unstable cluster is
to other stable clusters. If it is too dissimilar, it is kept as its own cluster. Otherwise,
the unstable cluster is dissolved and its cells are assigned to their best-matching
clusters.

It determines whether to dissolve a cluster by calculating whether or not each cell in the
unstable cluster has a good match to a stable cluster (i.e., the co-clustering rate exceeds
``coclust_threshold``). If enough of the cells of the unstable cluster have good matches
(the fraction of cells exceeds ``pct_needed``), the cluster is dissolved.

.. autoclass:: RefineParameters

.. autofunction:: stable_match_rates
.. autofunction:: match_rates_for_unstable_clusters
.. autofunction:: new_labels_for_dissolved_cluster

"""

import numpy as np
import pandas as pd
import argschema as ags
import logging


class RefineParameters(ags.ArgSchema):
    """Parameter schema for unstable cluster refinement"""
    cocluster_matrix_file = ags.fields.InputFile(
        description="File path for co-clustering matrix")
    jaccards_file = ags.fields.InputFile(
        description="File path for Jaccard scores")
    cluster_labels_file = ags.fields.InputFile(
        description="File path for cluster labels")

    refined_labels_file = ags.fields.OutputFile(
        description="Output file path for refined cluster labels (as integers)")
    refined_text_labels_file = ags.fields.OutputFile(
        description="Output file path for refined cluster labels (as strings with me_prefix)")
    refined_ordering_file = ags.fields.OutputFile(
        description="Output file path for refined cluster labels")
    unstable_threshold = ags.fields.Float(
        description="Threshold for Jaccard score to determine stability",
        default=0.5)
    coclust_threshold = ags.fields.Float(
        description="Threshold for co-clustering rate to be considered a match to another cluster",
        default=0.4)
    pct_needed = ags.fields.Float(
        description="Minimum fraction of matching cells to dissolve a cluster",
        default=0.33)
    me_prefix = ags.fields.String(
        description="prefix for refined cluster text labels")


def stable_match_rates(clust_labels, shared, stable_clusters):
    """Calculate the co-clustering rates of cells within the stable clusters

    Parameters
    ----------
    clust_labels : (n, ) array
        Cluster labels for the `n` samples
    shared : (n, n) array
        Co-clustering rates between all `n` samples
    stable_clusters : list
        List of labels of the stable clusters

    Returns
    -------
    list
        Returns list of the average within-cluster co-clustering rates for every cell
        found within stable clusters
    """

    rates = []
    for cl in stable_clusters:
        cl_mask = cl == clust_labels
        cl_cells = np.flatnonzero(cl_mask)
        for cell_index in cl_cells:
            my_mask = np.ones(shared.shape[0]).astype(bool)
            my_mask[cell_index] = False
            rates.append(np.mean(shared[cell_index, :][my_mask & cl_mask]))
    return rates


def match_rates_for_unstable_clusters(unstable_clusters, stable_clusters, clust_labels, shared, threshold):
    """Calculate the fraction of cells in unstable clusters that match to a stable cluster.

    The highest co-clustering rate with a stable cluster is calculated for each cell in
    an unstable cluster. If that rate exceeds ``threshold``, that cell is categorized as
    matching another cluster. The fraction of matching cells is returned for each
    unstable cluster.

    Parameters
    ----------
    unstable_clusters : list
        List of labels of the unstable clusters
    stable_clusters : list
        List of labels of the stable clusters
    clust_labels : (n, ) array
        Cluster labels for the `n` samples
    shared : (n, n) array
        Co-clustering rates between all `n` samples
    threshold : float
        Minimum co-clustering rate to be considered a match with another cluster

    Returns
    -------
    dict
        Dictionary of unstable clusters (keys) and their fractions of matching cells (values)
    """
    results = []
    for cl in unstable_clusters:
        cl_cells = np.flatnonzero(cl == clust_labels)
        if len(cl_cells) == 0:
            continue
        best_match_rates = []
        for cell_index in cl_cells:
            other_rates = []
            for other_cl in stable_clusters:
                other_mask = other_cl == clust_labels
                other_rates.append(np.mean(shared[cell_index, :][other_mask]))
            best_match_rates.append(np.max(other_rates))
        results.append((cl, np.sum(np.array(best_match_rates) > threshold).astype(float) / len(best_match_rates)))
    return dict(results)


def new_labels_for_dissolved_cluster(cl, clust_labels, shared, stable_clusters):
    """ Relabel the cells in a dissolved cluster with their new assignments

    Parameters
    ----------
    cl : int
        Cluster that will be dissolved
    clust_labels : (n, ) array
        Cluster labels for the `n` samples
    shared : (n, n) array
        Co-clustering rates between all `n` samples
    stable_clusters : list
        List of labels of the stable clusters

    Returns
    -------
    (n, ) array
        Array with updated cluster labels
    """

    cl_cells = np.flatnonzero(cl == clust_labels)
    new_labelling = []
    for cell_index in cl_cells:
        other_rates = []
        for other_cl in stable_clusters:
            other_mask = other_cl == clust_labels
            other_rates.append(np.mean(shared[cell_index, :][other_mask]))
        new_cl = stable_clusters[np.argmax(other_rates)]
        new_labelling.append((cell_index, new_cl))
    new_labels = clust_labels.copy()
    for ci, new_cl in new_labelling:
        new_labels[ci] = new_cl
    return new_labels


def main(cocluster_matrix_file, jaccards_file, cluster_labels_file,
         refined_labels_file, refined_text_labels_file, refined_ordering_file,
         unstable_threshold, coclust_threshold, pct_needed, me_prefix,
         **kwargs):
    """ Main runner function for script.

    See argschema input parameters for argument descriptions.
    """

    shared = np.loadtxt(cocluster_matrix_file)
    jaccards = np.loadtxt(jaccards_file)
    clust_labels_df = pd.read_csv(cluster_labels_file, index_col=0)
    print(clust_labels_df.head())
    clust_labels = clust_labels_df["0"].values

    unstable_clusters = np.flatnonzero(jaccards.mean(axis=1) < unstable_threshold)
    stable_clusters = np.flatnonzero(jaccards.mean(axis=1) >= unstable_threshold)
    logging.info("Found {:d} stable clusters".format(len(stable_clusters)))
    logging.info("Found {:d} unstable clusters".format(len(unstable_clusters)))

    stable_rates = stable_match_rates(clust_labels, shared, stable_clusters)

    logging.info("Using coclustering threshold of {:g}".format(coclust_threshold))
    logging.info("cf. stable rates 5th percentile {:g}".format(np.percentile(stable_rates, 5)))

    keep_going = True
    refined_clust_labels = clust_labels.copy()
    while keep_going:
        mr = match_rates_for_unstable_clusters(unstable_clusters, stable_clusters,
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
                                                                shared, stable_clusters)

    refined_order = np.lexsort((clust_labels, refined_clust_labels))
    refined_relabel = {v: i + 1 for i, v in enumerate(np.sort(np.unique(refined_clust_labels)))}
    refined_text_labels = ["{:s}_{:d}".format(me_prefix, refined_relabel[v]) for v in refined_clust_labels]

    pd.DataFrame(refined_clust_labels, index=clust_labels_df.index.values).to_csv(refined_labels_file)
    pd.DataFrame(refined_text_labels, index=clust_labels_df.index.values).to_csv(refined_text_labels_file)
    np.savetxt(refined_ordering_file, refined_order, fmt="%d")


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=RefineParameters)
    main(**module.args)
