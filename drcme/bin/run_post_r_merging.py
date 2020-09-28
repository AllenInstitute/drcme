"""
Script for determining how many GMM components to merge.

Gaussian mixture model (GMM) components are merged into a smaller number of clusters
using an entropy criterion as described by `Baudry et al. (2010) <https://www.tandfonline.com/doi/abs/10.1198/jcgs.2010.08111>`_.
A piecewise linear fit is used to determine the point at which merging should terminate.

.. autoclass:: PostGmmMergingParameters

"""

import drcme.post_gmm_merging as pgm
import pandas as pd
import os
import json
import argschema as ags

class PostGmmMergingParameters(ags.ArgSchema):
    """Parameter schema for merging"""
    tau_file = ags.fields.InputFile(
        description="Path to file with cluster membership probabilities")
    labels_file = ags.fields.InputFile(
        description="Path to file with cluster labels")
    merge_info_file = ags.fields.OutputFile(
        description="Path to JSON file with number of components after entropy-based merging")
    entropy_piecewise_components = ags.fields.Integer(
        default=3,
        description="Number of components (2 or 3) for piecewise linear fit of entropy scores",
        validate=lambda x: x in [2, 3])


def main(tau_file, labels_file, merge_info_file, entropy_piecewise_components, **kwargs):
    """ Main runner function for script.

    See :class:`PostGmmMergingParameters` parameters for argument descriptions.
    """
    tau = pd.read_csv(tau_file, index_col=0).values
    labels = pd.read_csv(labels_file, index_col=0).values
    K_bic = labels.max()
    labels -= 1
    merge_info, new_labels, tau_merged, merge_matrix = pgm.entropy_combi(tau, labels, K_bic, entropy_piecewise_components)

    with open(os.path.join(merge_info_file), "w") as f:
        json.dump({
            "gmm_components": int(K_bic),
            "postmerge_clusters": int(K_bic - merge_info["cp"][0]),
        }, f, indent=4)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=PostGmmMergingParameters)
    main(**module.args)
