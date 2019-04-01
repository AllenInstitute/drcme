import numpy as np
import pandas as pd
from drcme.spca_fit import spca_on_all_data, consolidate_spca
import drcme.load_data as ld
import argschema as ags
from sklearn.externals import joblib
import logging
import os
import json


class DatasetParameters(ags.schemas.DefaultSchema):
    project = ags.fields.String(default="T301")
    data_dir = ags.fields.InputDir(default="/allen/programs/celltypes/workgroups/ivscc/nathang/single-cell-ephys/single_cell_ephys")
    dendrite_type = ags.fields.String(default="all", validate=lambda x: x in ["all", "spiny", "aspiny"])
    allow_missing_structure = ags.fields.Boolean(required=False, default=False)
    allow_missing_dendrite = ags.fields.Boolean(required=False, default=False)
    limit_to_cortical_layers = ags.fields.List(ags.fields.String, default=[], cli_as_single_argument=True)


class AnalysisParameters(ags.ArgSchema):
    params_file = ags.fields.InputFile(default="/allen/programs/celltypes/workgroups/ivscc/nathang/single-cell-ephys/dev/default_spca_params.json")
    output_dir = ags.fields.OutputDir(description="directory for output files")
    output_code = ags.fields.String(description="code for output files")
    use_noise = ags.fields.Boolean(required=False, default=False, description="use noise features if true")
    datasets = ags.fields.Nested(DatasetParameters,
                                 required=True,
                                 many=True,
                                 description="schema for loading one or more specific datasets for the analysis")


def main(params_file, output_dir, output_code, use_noise, datasets, **kwargs):

    # Load data from each dataset
    data_objects = []
    specimen_ids_list = []
    for ds in datasets:
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
        data_objects.append(data_for_spca)
        specimen_ids_list.append(specimen_ids)

    data_for_spca = []
    for i, do in enumerate(data_objects):
        for j, data_item in enumerate(do):
            if i == 0:
                data_for_spca.append({
                    "data": data_item["data"].copy(),
                    "part_keys": data_item["part_keys"],
                })
            else:
                data_for_spca[j]["data"] = np.vstack([data_for_spca[j]["data"],
                                                      data_item["data"]])
    specimen_ids = np.hstack(specimen_ids_list)

    if len(specimen_ids) != data_for_spca[0]["data"].shape[0]:
        logging.error("Mismatch of specimen id dimension ({:d}) and data dimension ({:d})".format(len(specimen_ids), data_for_spca[0]["data"].shape[0]))

    logging.info("Proceeding with %d cells", len(specimen_ids))

    # Load parameters
    spca_zht_params, _ = ld.define_spca_parameters(filename=params_file)

    # Run sPCA
    spca_results = spca_on_all_data(data_for_spca, spca_zht_params)

    logging.info("Saving results...")
    joblib.dump(spca_results, os.path.join(output_dir, "spca_loadings_{:s}.pkl".format(output_code)))
    combo, component_record = consolidate_spca(spca_results, data_for_spca, spca_zht_params)
    combo_df = pd.DataFrame(combo, index=specimen_ids)
    combo_df.to_csv(os.path.join(output_dir, "sparse_pca_components_{:s}.csv".format(output_code)))
    with open(os.path.join(output_dir, "spca_components_used_{:s}.json".format(output_code)), "w") as f:
        json.dump(component_record, f, indent=4)
    logging.info("Done.")


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=AnalysisParameters)
    main(**module.args)
