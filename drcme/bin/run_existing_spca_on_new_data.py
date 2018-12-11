import pandas as pd
import argschema as ags
from sklearn.externals import joblib
import drcme.load_data as ld
from drcme.spca_transform import orig_mean_and_std_for_zscore, spca_transform_new_data
import logging


class SpcaTransformParameters(ags.ArgSchema):
    orig_transform_file = ags.fields.InputFile(default="/allen/programs/celltypes/workgroups/single-cell-ephys/dev/spca_loadings_T301.pkl")
    orig_data_dir = ags.fields.InputDir(default="/allen/programs/celltypes/workgroups/single-cell-ephys/single_cell_ephys/")
    orig_project = ags.fields.String(default="T301")
    new_data_dir = ags.fields.InputDir()
    new_project = ags.fields.String()
    params_file = ags.fields.InputFile(default="/allen/aibs/mat/nathang/single-cell-ephys/dev/default_spca_params.json")
    dendrite_type = ags.fields.String(default="all", validate=lambda x: x in ["all", "spiny", "aspiny"])
    output_file = ags.fields.OutputFile()


def main(orig_transform_file, orig_data_dir, orig_project,
         new_data_dir, new_project,
         params_file, dendrite_type,
         output_file, **kwargs):
    spca_zht_params = ld.define_spca_parameters(params_file)

    spca_results = joblib.load(orig_transform_file)

    # These arguments should be parameterized
    orig_dataset, orig_ids = ld.load_organized_data(project=orig_project,
                                          base_dir=orig_data_dir,
                                          params_file=params_file,
                                          dendrite_type=dendrite_type,
                                          use_noise=False,
                                          need_structure=False,
                                          include_dend_type_null=False)
    logging.info("Original dataset had {:d} cells".format(len(orig_ids)))
    orig_mean, orig_std = orig_mean_and_std_for_zscore(spca_results, orig_dataset, spca_zht_params)

    data_for_spca, new_ids = ld.load_organized_data(project=new_project,
                                           base_dir=new_data_dir,
                                           params_file=params_file,
                                           dendrite_type=dendrite_type,
                                           use_noise=False,
                                           need_structure=False,
                                           include_dend_type_null=True)
    logging.info("Applying transform to {:d} new cells".format(len(new_ids)))
    new_combo = spca_transform_new_data(spca_results,
                                        data_for_spca,
                                        spca_zht_params,
                                        orig_mean, orig_std)
    new_combo_df = pd.DataFrame(new_combo, index=new_ids)
    new_combo_df.to_csv(output_file)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=SpcaTransformParameters)
    main(**module.args)