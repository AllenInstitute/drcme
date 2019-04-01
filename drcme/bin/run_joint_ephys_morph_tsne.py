import pandas as pd
import argschema as ags
import drcme.tsne as tsne


class JointTsneParameters(ags.ArgSchema):
    ephys_file_1 = ags.fields.InputFile()
    ephys_file_2 = ags.fields.InputFile()
    morph_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()
    relative_ephys_weight = ags.fields.Float(default=1.)
    n_components = ags.fields.Integer(default=2)
    perplexity = ags.fields.Float(default=25.)
    n_iter = ags.fields.Integer(default=20000)


def main(ephys_file_1, ephys_file_2, morph_file, output_file, relative_ephys_weight,
         n_components, perplexity, n_iter, **kwargs):
    ephys_df_1 = pd.read_csv(ephys_file_1, index_col=0)
    ephys_df_2 = pd.read_csv(ephys_file_2, index_col=0)
    ephys_df = pd.concat([ephys_df_1, ephys_df_2])
    morph_df = pd.read_csv(morph_file, index_col=0).dropna(axis=1)

    result = tsne.dual_modal_tsne(ephys_df, morph_df,relative_ephys_weight,
                                  n_components, perplexity, n_iter)
    result.to_csv(output_file)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=JointTsneParameters)
    main(**module.args)
