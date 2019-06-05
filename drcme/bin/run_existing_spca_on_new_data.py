import numpy as np
import pandas as pd
import argschema as ags
from sklearn.externals import joblib
import drcme.load_data as ld
from drcme.spca_transform import orig_mean_and_std_for_zscore, spca_transform_new_data
import logging


class DatasetParameters(ags.schemas.DefaultSchema):
    project = ags.fields.String(default="T301")
    data_dir = ags.fields.InputDir(default="/allen/programs/celltypes/workgroups/ivscc/nathang/single-cell-ephys/single_cell_ephys")
    dendrite_type = ags.fields.String(default="all", validate=lambda x: x in ["all", "spiny", "aspiny"])
    allow_missing_structure = ags.fields.Boolean(required=False, default=False)
    allow_missing_dendrite = ags.fields.Boolean(required=False, default=False)
    limit_to_cortical_layers = ags.fields.List(ags.fields.String, default=[], cli_as_single_argument=True)


class SpcaTransformParameters(ags.ArgSchema):
    orig_transform_file = ags.fields.InputFile(description="sPCA loadings file")
    orig_datasets = ags.fields.Nested(DatasetParameters,
        required=True,
        many=True,
        description="schema for loading one or more specific datasets for the analysis")
    new_datasets = ags.fields.Nested(DatasetParameters,
        required=True,
        many=True,
        description="schema for loading one or more specific datasets for the analysis")
    params_file = ags.fields.InputFile(default="/allen/aibs/mat/nathang/single-cell-ephys/dev/default_spca_params.json")
    output_file = ags.fields.OutputFile(description="CSV with transformed values")
    use_noise = ags.fields.Boolean(default=False)


def main(orig_transform_file, orig_datasets, new_datasets, params_file,
         output_file, use_noise, **kwargs):
    spca_zht_params, _ = ld.define_spca_parameters(params_file)

    spca_results = joblib.load(orig_transform_file)

    # These arguments should be parameterized
    orig_data_objects = []
    orig_specimen_ids_list = []
    for ds in orig_datasets:
        if len(ds["limit_to_cortical_layers"]) == 0:
            limit_to_cortical_layers = None
        else:
            limit_to_cortical_layers = ds["limit_to_cortical_layers"]

        data_for_spca, specimen_ids = ld.load_organized_data(project=ds["project"],
                                            base_dir=ds["data_dir"],
                                            use_noise=use_noise,
                                            dendrite_type=ds["dendrite_type"],
                                            need_structure=not ds["allow_missing_structure"],
                                            include_dend_type_null=ds["allow_missing_dendrite"],
                                            limit_to_cortical_layers=limit_to_cortical_layers,
                                            params_file=params_file)
        orig_data_objects.append(data_for_spca)
        orig_specimen_ids_list.append(specimen_ids)
    orig_data_for_spca = []
    for i, do in enumerate(orig_data_objects):
        for j, data_item in enumerate(do):
            if i == 0:
                orig_data_for_spca.append({
                    "data": data_item["data"].copy(),
                    "part_keys": data_item["part_keys"],
                })
            else:
                orig_data_for_spca[j]["data"] = np.vstack([orig_data_for_spca[j]["data"],
                                                      data_item["data"]])
    orig_specimen_ids = np.hstack(orig_specimen_ids_list)
    logging.info("Original datasets had {:d} cells".format(len(orig_specimen_ids)))
    orig_mean, orig_std = orig_mean_and_std_for_zscore(spca_results, orig_data_for_spca, spca_zht_params)

    new_data_objects = []
    new_specimen_ids_list = []
    for ds in new_datasets:
        if len(ds["limit_to_cortical_layers"]) == 0:
            limit_to_cortical_layers = None
        else:
            limit_to_cortical_layers = ds["limit_to_cortical_layers"]

        data_for_spca, specimen_ids = ld.load_organized_data(project=ds["project"],
                                            base_dir=ds["data_dir"],
                                            use_noise=use_noise,
                                            dendrite_type=ds["dendrite_type"],
                                            need_structure=not ds["allow_missing_structure"],
                                            include_dend_type_null=ds["allow_missing_dendrite"],
                                            limit_to_cortical_layers=limit_to_cortical_layers,
                                            params_file=params_file)
        new_data_objects.append(data_for_spca)
        new_specimen_ids_list.append(specimen_ids)
    data_for_spca = []
    for i, do in enumerate(new_data_objects):
        for j, data_item in enumerate(do):
            if i == 0:
                data_for_spca.append({
                    "data": data_item["data"].copy(),
                    "part_keys": data_item["part_keys"],
                })
            else:
                data_for_spca[j]["data"] = np.vstack([data_for_spca[j]["data"],
                                                      data_item["data"]])
    new_ids = np.hstack(new_specimen_ids_list)
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