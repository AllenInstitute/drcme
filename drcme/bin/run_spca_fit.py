import pandas as pd
from drcme.spca_fit import spca_on_all_data, consolidate_spca
import drcme.load_data as ld
import argschema as ags
from sklearn.externals import joblib
import logging
import os
import json


class AnalysisParameters(ags.ArgSchema):
    project = ags.fields.String(default="T301")
    data_dir = ags.fields.InputDir(default="/allen/programs/celltypes/workgroups/ivscc/nathang/single-cell-ephys/single_cell_ephys")
    params_file = ags.fields.InputFile(default="/allen/programs/celltypes/workgroups/ivscc/nathang/single-cell-ephys/dev/default_spca_params.json")
    output_dir = ags.fields.OutputDir()
    dendrite_type = ags.fields.String(default="all", validate=lambda x: x in ["all", "spiny", "aspiny"])
    use_noise = ags.fields.Boolean(required=False, default=False)
    allow_missing_structure = ags.fields.Boolean(required=False, default=False)
    allow_missing_dendrite = ags.fields.Boolean(required=False, default=False)


def main(project, data_dir, params_file, output_dir, dendrite_type, use_noise,
         allow_missing_structure, allow_missing_dendrite, **kwargs):

    # Load data
    data_for_spca, specimen_ids = ld.load_organized_data(project=project,
                                        base_dir=data_dir,
                                        use_noise=use_noise,
                                        dendrite_type=dendrite_type,
                                        need_structure=not allow_missing_structure,
                                        include_dend_type_null=allow_missing_dendrite,
                                        params_file=params_file)
    logging.info("Proceeding with %d cells", len(specimen_ids))

    # Load parameters
    spca_zht_params = ld.define_spca_parameters(filename=params_file)

    # Run sPCA
    spca_results = spca_on_all_data(data_for_spca, spca_zht_params)

    logging.info("Saving results...")
    joblib.dump(spca_results, os.path.join(output_dir, "spca_loadings_{:s}.pkl".format(project)))
    combo, component_record = consolidate_spca(spca_results, data_for_spca, spca_zht_params)
    combo_df = pd.DataFrame(combo, index=specimen_ids)
    combo_df.to_csv(os.path.join(output_dir, "sparse_pca_components_{:s}.csv".format(project)))
    with open(os.path.join(output_dir, "spca_components_used_{:s}.json".format(project)), "w") as f:
        json.dump(component_record, f, indent=4)
    logging.info("Done.")

if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=AnalysisParameters)
    main(**module.args)
