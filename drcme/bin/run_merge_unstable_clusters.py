"""
Script for merging unstable clusters into stable ones.

.. autoclass:: UnstableClusterMergingParameters

"""

import drcme.post_gmm_merging as pgm
import numpy as np
import pandas as pd
import os
import json
import logging
import argschema as ags

class UnstableClusterMergingParameters(ags.ArgSchema):
    """Parameter schema for unstable cluster merging"""
    components_file = ags.fields.InputFile(
        description="Path to CSV file with sPCA components")
    post_merge_tau_file = ags.fields.InputFile(
        description="Path to file with cluster membership probabilities after entropy-based merging")
    post_merge_labels_file = ags.fields.InputFile(
        description="Path to file with cluster labels after entropy-based merging")
    jaccard_file = ags.fields.InputFile(
        description="Path to file with Jaccard index values after subset-based stability analysis")
    post_merge_proba_file = ags.fields.OutputFile(
        description="Path to file with post-merging cluster membership probabilities")
    etypes_file = ags.fields.OutputFile(
        description="Path to file with stable e-type assignments")
    merge_unstable_info_file = ags.fields.OutputFile(
        description="Path to file with merge sequence information")
    stability_threshold = ags.fields.Float(
        description="Threshold below which clusters are considered unstable",
        default=0.5)
    outliers = ags.fields.List(ags.fields.Integer,
        description="Specimen IDs to exclude from analysis"
    )


def main(components_file, post_merge_tau_file, post_merge_labels_file, jaccard_file,
         post_merge_proba_file, etypes_file, merge_unstable_info_file,
         outliers, stability_threshold, **kwargs):
    """ Main runner function for script.

    See :class:`UnstableClusterMergingParameters` for argument descriptions.
    """

    data = pd.read_csv(components_file, index_col=0)
    tau = pd.read_csv(post_merge_tau_file, index_col=0)
    labels = pd.read_csv(post_merge_labels_file, index_col=0)
    jaccard_means = pd.read_csv(jaccard_file, index_col=0)

    K_bic = labels.values.max()

    labels -= 1 # shift from R to Python indexing
    specific_merges = np.flatnonzero(jaccard_means.values < stability_threshold) # definition of instability
    (merge_info,
     merge_labels,
     tau_merged,
     merge_matrix) = pgm.entropy_specific_merges(tau=tau.values,
                                                 labels=labels.values,
                                                 K_bic=K_bic,
                                                 clusters_to_merge=specific_merges)

    merge_labels_reorder, tau_merged, _, _ = pgm.order_new_labels(merge_labels, tau_merged, data.loc[~data.index.isin(outliers), :])
    indexes = data.loc[~data.index.isin(outliers), :].index.values

    np.savetxt(post_merge_proba_file, tau_merged)
    pd.DataFrame(merge_labels_reorder, index=indexes).to_csv(etypes_file)
    with open(merge_unstable_info_file, "w") as f:
        json.dump({"merges": merge_info["merges_by_names"]}, f)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=UnstableClusterMergingParameters)
    main(**module.args)
