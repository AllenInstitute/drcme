#!/usr/bin/env python

import numpy as np
import pandas as pd
import drcme.ephys_morph_clustering as emc
import argschema as ags
import logging
import sys


class MeClusteringParameters(ags.ArgSchema):
    ephys_file = ags.fields.InputFile()
    morph_file = ags.fields.InputFile()
    weights = ags.fields.List(ags.fields.Float,
                              cli_as_single_argument=True,
                              default=[1., 2., 4.])
    n_cl = ags.fields.List(ags.fields.Integer,
        cli_as_single_argument=True,
        default=[10, 15, 20, 25])
    min_consensus_n = ags.fields.Integer(default=3)
    cocluster_matrix_file = ags.fields.OutputFile()
    cluster_labels_file = ags.fields.OutputFile()
    specimen_id_file = ags.fields.OutputFile()
    jaccards_file = ags.fields.OutputFile()
    ordering_file = ags.fields.OutputFile()


def main(ephys_file, morph_file,
         weights, n_cl, min_consensus_n, cocluster_matrix_file,
         cluster_labels_file, jaccards_file, ordering_file,
         specimen_id_file,
         **kwargs):
    # Load the data
    ephys_data = pd.read_csv(ephys_file, index_col=0)

    # Expect already normalized wide dataframe
    morph_data = pd.read_csv(morph_file, index_col=0)
    morph_ids = morph_data.index.values

    # Use cells with both types of data
    ephys_morph_ids = ephys_data.index.intersection(morph_data.index)

    logging.info(f"Using {len(ephys_morph_ids)} cells")

    logging.info("Calculating cluster calls")
    logging.info("Ephys weights: " + ", ".join(map(str, weights)))
    logging.info("Cluster numbers: " + ", ".join(map(str, n_cl)))

    results_df = emc.all_cluster_calls(ephys_morph_ids.values,
                                       morph_data.loc[ephys_morph_ids, :].values,
                                       ephys_data.loc[ephys_morph_ids, :].values,
                                       weights=weights,
                                       n_cl=n_cl)
    clust_labels, shared, cc_rates = emc.consensus_clusters(
        results_df.values[:, 1:], min_clust_size=min_consensus_n)
    new_order = emc.sort_order(clust_labels)

    logging.info(f"Identified {len(np.unique(clust_labels))} consensus clusters with full data set")

    np.savetxt(cocluster_matrix_file, shared)
    pd.DataFrame(clust_labels, index=ephys_morph_ids.values).to_csv(cluster_labels_file)
    np.savetxt(ordering_file, new_order, fmt="%d")
    np.savetxt(specimen_id_file, ephys_morph_ids.values, fmt="%d")

    logging.info("Evaluating cluster stability")
    jaccards = emc.subsample_run(clust_labels,
                                 ephys_morph_ids.values,
                                 morph_data.loc[ephys_morph_ids, :].values,
                                 ephys_data.loc[ephys_morph_ids, :].values,
                                 weights=weights,
                                 n_cl=n_cl,
                                 n_folds=10,
                                 n_iter=10,
                                 min_consensus_n=min_consensus_n)
    np.savetxt(jaccards_file, jaccards)

    logging.info("Done")


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=MeClusteringParameters)
    main(**module.args)