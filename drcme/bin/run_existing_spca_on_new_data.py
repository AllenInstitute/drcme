"""
Script to apply an existing set of sPCA loadings to a new data set.

.. autoclass:: DatasetParameters
.. autoclass:: SpcaTransformParameters

"""

import numpy as np
import pandas as pd
import argschema as ags
import joblib
import drcme.load_data as ld
from drcme.spca import orig_mean_and_std_for_zscore, spca_transform_new_data
import logging


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


class SpcaTransformParameters(ags.ArgSchema):
    """Parameter schema for sPCA using existing transform"""
    orig_transform_file = ags.fields.InputFile(description="sPCA loadings file")
    orig_datasets = ags.fields.Nested(DatasetParameters,
        required=True,
        many=True,
        description="schema for loading one or more specific datasets for the analysis")
    new_datasets = ags.fields.Nested(DatasetParameters,
        required=True,
        many=True,
        description="schema for loading one or more specific datasets for the analysis")
    params_file = ags.fields.InputFile(
        description="JSON file with sPCA parameters")
    output_file = ags.fields.OutputFile(description="CSV with transformed values")


def main(orig_transform_file, orig_datasets, new_datasets, params_file,
         output_file, **kwargs):
    """ Main runner function for script.

    See :class:`SpcaTransformParameters` for argument descriptions.
    """

    spca_zht_params, _ = ld.define_spca_parameters(params_file)

    spca_results = joblib.load(orig_transform_file)

    # Load original data sets
    orig_data_objects = []
    orig_specimen_ids_list = []
    for ds in orig_datasets:
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
        orig_data_objects.append(data_for_spca)
        orig_specimen_ids_list.append(specimen_ids)
    orig_data_for_spca = []
    for i, do in enumerate(orig_data_objects):
        for k in do:
            if k not in orig_data_for_spca:
                orig_data_for_spca[k] = do[k]
            else:
                orig_data_for_spca[k] = np.vstack([orig_data_for_spca[k], do[k]])
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

        data_for_spca, specimen_ids = ld.load_h5_data(h5_fv_file=ds["fv_h5_file"],
                                            metadata_file=ds["metadata_file"],
                                            dendrite_type=ds["dendrite_type"],
                                            need_structure=not ds["allow_missing_structure"],
                                            need_ramp_spike = ds["need_ramp_spike"],
                                            include_dend_type_null=ds["allow_missing_dendrite"],
                                            limit_to_cortical_layers=limit_to_cortical_layers,
                                            id_file=ds["id_file"],
                                            params_file=params_file)
        new_data_objects.append(data_for_spca)
        new_specimen_ids_list.append(specimen_ids)
    data_for_spca = []
    for i, do in enumerate(new_data_objects):
        for k in do:
            if k not in data_for_spca:
                data_for_spca[k] = do[k]
            else:
                data_for_spca[k] = np.vstack([data_for_spca[k], do[k]])

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