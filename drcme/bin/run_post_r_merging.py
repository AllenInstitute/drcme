#!/usr/bin/env python

import drcme.post_gmm_merging as pgm
import pandas as pd
import os
import json
import argschema as ags

class PostGmmMergingParameters(ags.ArgSchema):
    tau_file = ags.fields.InputFile()
    labels_file = ags.fields.InputFile()
    merge_info_file = ags.fields.OutputFile()
    entropy_piecewise_components = ags.fields.Integer(default=3, validate=lambda x: x in [2, 3])


def main(tau_file, labels_file, merge_info_file, entropy_piecewise_components, **kwargs):
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
