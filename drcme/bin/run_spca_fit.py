"""
Script to run sparse principal component analysis on electrophysiology
feature vectors.

The electrophysiology feature vectors used as inputs are in the form
processed by the `IPFX package <https://ipfx.readthedocs.io>`_. An
optional metadata file can be given as an input to filter the cells in
the data set.

The script produces several outputs in the specified ``output_dir``,
named with a specified ``output_code``.

- *sparse_principal_components_[output_code].csv*: sPCA values for each
  cell.
- *spca_components_used_[output_code].json*: List of kept components
  for each data subset.
- *spca_loadings_[output_code].pkl*: sPCA loadings, adjust explained
  variance, and transformed values for each data subset. Uses the
  ``joblib`` library for saving/loading.

.. autoclass:: DatasetParameters
.. autoclass:: AnalysisParameters

"""

import numpy as np
import pandas as pd
import drcme.spca as spca
import drcme.load_data as ld
import argschema as ags
import joblib
import logging
import os
import json


class DatasetParameters(ags.schemas.DefaultSchema):
    """Parameter schema for input datasets"""
    fv_h5_file = ags.fields.InputFile(
        description="HDF5 file with feature vectors")
    metadata_file = ags.fields.InputFile(
        description="Metadata file in CSV format",
        allow_none=True,
        default=None)
    dendrite_type = ags.fields.String(
        default="all",
        description="Filter for dendrite type using information in metadata (all, spiny, aspiny)",
        validate=lambda x: x in ["all", "spiny", "aspiny"])
    allow_missing_structure = ags.fields.Boolean(
        required=False,
        description="Whether or not structure value for cell in metadata can be missing",
        default=False)
    allow_missing_dendrite = ags.fields.Boolean(
        required=False,
        description="Whether or not dendrite type value for cell in metadata can be missing",
        default=False)
    need_ramp_spike = ags.fields.Boolean(
        required=False,
        description="Whether or not to exclude cells that did not fire an action potential from the ramp stimulus",
        default=True)
    limit_to_cortical_layers = ags.fields.List(
        ags.fields.String,
        description="List of cortical layers to limit the data set (using the metadata file)",
        default=[],
        cli_as_single_argument=True)
    id_file = ags.fields.InputFile(
        description="Text file with specimen IDs to use. Cells with IDs not in the file will be excluded.",
        required=False,
        allow_none=True,
        default=None)


class AnalysisParameters(ags.ArgSchema):
    """Parameter schema for sPCA analysis"""
    params_file = ags.fields.InputFile(
        description="JSON file with sPCA parameters")
    output_dir = ags.fields.OutputDir(
        description="Directory for output files")
    output_code = ags.fields.String(
        description="Code for naming output files")
    datasets = ags.fields.Nested(DatasetParameters,
                                 required=True,
                                 many=True,
                                 description="Schema for loading one or more specific datasets for the analysis")


def main(params_file, output_dir, output_code, datasets, **kwargs):
    """ Main runner function for script.

    See argschema input parameters for argument descriptions.
    """

    # Load data from each dataset
    data_objects = []
    specimen_ids_list = []
    for ds in datasets:
        if len(ds["limit_to_cortical_layers"]) == 0:
            limit_to_cortical_layers = None
        else:
            limit_to_cortical_layers = ds["limit_to_cortical_layers"]

        data_for_spca, specimen_ids = ld.load_h5_data(h5_fv_file=ds["fv_h5_file"],
                                            metadata_file=ds["metadata_file"],
                                            dendrite_type=ds["dendrite_type"],
                                            need_structure=not ds["allow_missing_structure"],
                                            need_ramp_spike = ds["need_ramp_spike"],
                                            include_dend_type_null=ds["allow_missing_dendrite"],
                                            limit_to_cortical_layers=limit_to_cortical_layers,
                                            id_file=ds["id_file"],
                                            params_file=params_file)
        data_objects.append(data_for_spca)
        specimen_ids_list.append(specimen_ids)

    data_for_spca = {}
    for i, do in enumerate(data_objects):
        for k in do:
            if k not in data_for_spca:
                data_for_spca[k] = do[k]
            else:
                data_for_spca[k] = np.vstack([data_for_spca[k], do[k]])
    specimen_ids = np.hstack(specimen_ids_list)

    first_key = list(data_for_spca.keys())[0]
    if len(specimen_ids) != data_for_spca[first_key].shape[0]:
        logging.error("Mismatch of specimen id dimension ({:d}) and data dimension ({:d})".format(len(specimen_ids), data_for_spca[first_key].shape[0]))

    logging.info("Proceeding with %d cells", len(specimen_ids))

    # Load parameters
    spca_zht_params, _ = ld.define_spca_parameters(filename=params_file)

    # Run sPCA
    subset_for_spca = spca.select_data_subset(data_for_spca, spca_zht_params)
    spca_results = spca.spca_on_all_data(subset_for_spca, spca_zht_params)
    combo, component_record = spca.consolidate_spca(spca_results)

    logging.info("Saving results...")
    joblib.dump(spca_results, os.path.join(output_dir, "spca_loadings_{:s}.pkl".format(output_code)))
    combo_df = pd.DataFrame(combo, index=specimen_ids)
    combo_df.to_csv(os.path.join(output_dir, "sparse_pca_components_{:s}.csv".format(output_code)))
    with open(os.path.join(output_dir, "spca_components_used_{:s}.json".format(output_code)), "w") as f:
        json.dump(component_record, f, indent=4)
    logging.info("Done.")


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=AnalysisParameters)
    main(**module.args)
