"""
Script to generate t-SNE coordinates using electrophysiology and morphological features.

The electrophysiology file is typically an sPCA value file. Both files should have
normalized features (e.g., z-scored features). The two data sets can be weighted differently
via the ``relative_ephys_weight`` parameter.

.. autoclass:: JointTsneParameters

"""

import pandas as pd
import argschema as ags
import drcme.tsne as tsne


class JointTsneParameters(ags.ArgSchema):
    """Parameter schema for joint t-SNE calculation"""

    ephys_file = ags.fields.InputFile(
        description="Path to electrophysiology data file")
    morph_file = ags.fields.InputFile(
        description="Path to morphology data file")
    output_file = ags.fields.OutputFile(
        description="Path to output file for t-SNE coordinates")
    relative_ephys_weight = ags.fields.Float(default=1.,
        description="Relative weight of electrophysiology values (vs morphology values)")
    perplexity = ags.fields.Float(default=25.,
        description="Perplexity parameter for t-SNE")
    n_iter = ags.fields.Integer(default=20000,
        description="Number of iterations for t-SNE")


def main(ephys_file, morph_file, output_file, relative_ephys_weight,
         perplexity, n_iter, **kwargs):
    """ Main runner function for script.

    See :class:`JointTsneParameters` for argument descriptions.
    """

    ephys_df = pd.read_csv(ephys_file, index_col=0)
    morph_df = pd.read_csv(morph_file, index_col=0).dropna(axis=1)

    result = tsne.dual_modal_tsne(ephys_df, morph_df, relative_ephys_weight,
                                  perplexity, n_iter)
    result.to_csv(output_file)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=JointTsneParameters)
    main(**module.args)
