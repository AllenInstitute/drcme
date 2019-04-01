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
    cocluster_matrix_file = ags.fields.OutputFile()
    cluster_labels_file = ags.fields.OutputFile()
    specimen_id_file = ags.fields.OutputFile()
    jaccards_file = ags.fields.OutputFile()
    ordering_file = ags.fields.OutputFile()


def main(ephys_file, morph_file,
         weights, cocluster_matrix_file,
         cluster_labels_file, jaccards_file, ordering_file,
         specimen_id_file,
         **kwargs):
    # Load the data
    ephys_data = pd.read_csv(ephys_file, index_col=0)

    # Expect already normalized wide dataframe
    morph_data = pd.read_csv(morph_file, index_col=0)
    morph_ids = morph_data.index.values

    # Get ephys data for cells with morphologies
    ids_with_morph_for_ephys = [s for s in morph_ids if s in ephys_data.index.tolist()]
    ephys_for_morph = ephys_data.loc[ids_with_morph_for_ephys, :]

    # Only use morphs that have ephys
    mask = [s in ephys_data.index.tolist() for s in morph_ids]
    morph_data_ephys = morph_data.loc[mask, :]

    logging.info("Calculating cluster calls")
    results_df = emc.all_cluster_calls(morph_data_ephys.index.values,
                                       morph_data_ephys.values,
                                       ephys_for_morph,
                                       weights=weights)
    clust_labels, shared, cc_rates = emc.consensus_clusters(results_df.values[:, 1:])
    new_order = emc.sort_order(clust_labels)
    np.savetxt(cocluster_matrix_file, shared)
    pd.DataFrame(clust_labels, index=morph_data_ephys.index.values).to_csv(cluster_labels_file)
    np.savetxt(ordering_file, new_order, fmt="%d")
    np.savetxt(specimen_id_file, morph_data_ephys.index.values, fmt="%d")

    logging.info("Evaluating cluster stability")
    jaccards = emc.subsample_run(clust_labels,
                                 morph_data_ephys.index.values,
                                 morph_data_ephys.values,
                                 ephys_for_morph,
                                 weights=weights,
                                 n_folds=10, n_iter=10)
    np.savetxt(jaccards_file, jaccards)

    logging.info("Done")


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=MeClusteringParameters)
    main(**module.args)