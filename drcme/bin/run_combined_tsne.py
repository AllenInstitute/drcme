import pandas as pd
import argschema as ags
import drcme.tsne as tsne


class ComboTsneParameters(ags.ArgSchema):
    spca_file_1 = ags.fields.InputFile()
    spca_file_2 = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()
    n_components = ags.fields.Integer(default=2)
    perplexity = ags.fields.Float(default=25.)
    n_iter = ags.fields.Integer(default=20000)


def main(spca_file_1, spca_file_2, output_file,
         n_components, perplexity, n_iter, **kwargs):
    df_1 = pd.read_csv(spca_file_1, index_col=0)
    df_2 = pd.read_csv(spca_file_2, index_col=0)

    combo_df = tsne.combined_tsne(df_1, df_2, n_components, perplexity, n_iter)
    combo_df.to_csv(output_file)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=ComboTsneParameters)
    main(**module.args)
