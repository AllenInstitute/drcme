"""
Script to generate t-SNE coordinates using two electrophysiology sPCA files.

The two files should be calculated using the same loadings. This is just a convenience
script to merge the two data sets, run t-SNE, and save the coordinates.

.. autoclass:: ComboTsneParameters

"""

import pandas as pd
import argschema as ags
import drcme.tsne as tsne


class ComboTsneParameters(ags.ArgSchema):
    """Parameter schema for combined t-SNE calculation"""

    spca_file_1 = ags.fields.InputFile(
        description="Path to first sPCA values file")
    spca_file_2 = ags.fields.InputFile(
        description="Path to second sPCA values file")
    output_file = ags.fields.OutputFile(
        description="Path to output file for t-SNE coordinates")
    n_components = ags.fields.Integer(default=2,
        description="Number of components for t-SNE")
    perplexity = ags.fields.Float(default=25.,
        description="Perplexity parameter for t-SNE")
    n_iter = ags.fields.Integer(default=20000,
        description="Number of iterations for t-SNE")


def main(spca_file_1, spca_file_2, output_file,
         n_components, perplexity, n_iter, **kwargs):
    """ Main runner function for script.

    See :class:`ComboTsneParameters` for argument descriptions.
    """

    df_1 = pd.read_csv(spca_file_1, index_col=0)
    df_2 = pd.read_csv(spca_file_2, index_col=0)

    combo_df = tsne.combined_tsne(df_1, df_2, n_components, perplexity, n_iter)
    combo_df.to_csv(output_file)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=ComboTsneParameters)
    main(**module.args)
