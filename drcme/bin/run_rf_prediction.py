import numpy as np
import pandas as pd
import drcme.prediction as pred
import argschema as ags
import logging


class RfPredictionParameters(ags.ArgSchema):
    reference_ephys_file = ags.fields.InputFile()
    prediction_ephys_file = ags.fields.InputFile()
    reference_label_file = ags.fields.InputFile()
    label_key = ags.fields.String()
    morph_file = ags.fields.InputFile(default=None, allow_none=True)
    output_file = ags.fields.OutputFile()
    ref_id_file = ags.fields.InputFile(default=None, allow_none=True)
    pred_id_file = ags.fields.InputFile(default=None, allow_none=True)
    n_trees = ags.fields.Integer(default=500)
    class_weight = ags.fields.String(default=None, allow_none=True)


def main(reference_ephys_file, prediction_ephys_file, reference_label_file,
         label_key, morph_file, output_file, ref_id_file, pred_id_file,
         n_trees, class_weight, **kwargs):
    ephys_ref = pd.read_csv(reference_ephys_file, index_col=0)
    ephys_pred = pd.read_csv(prediction_ephys_file, index_col=0)

    logging.debug("Running RF")
    ref_label_df = pd.read_csv(reference_label_file, index_col=0).set_index("specimen_id")

    if morph_file is None:
        ref_df = ephys_ref
        test_df = ephys_pred
    else:
        morph_df = pd.read_csv(morph_file, index_col=0)
        ref_df, test_df = construct_datasets(ephys_ref, ephys_pred, morph_df)

    if ref_id_file is not None:
        ref_ids = np.loadtxt(ref_id_file)
        ref_df = ref_df.loc[ref_ids, :]
    if pred_id_file is not None:
        pred_ids = np.loadtxt(pred_id_file)
        test_df = test_df.loc[pred_ids, :]

    labels = ref_label_df.loc[ref_df.index, label_key].values

    # drop reference values that don't have labels
    nan_mask = ~pd.Series(labels).isnull().values
    labels = labels[nan_mask]
    ref_df = ref_df.loc[nan_mask, :]

    pred_labels = pred.rf_predict(ref_df, labels, test_df,
                                  n_trees=n_trees, class_weight=class_weight)


    logging.debug("Saving results")
    pd.DataFrame(pred_labels, index=test_df.index.values).to_csv(output_file)


def construct_datasets(ephys_ref, ephys_pred, morph_df):
    ref_df = intersect_ephys_morph(ephys_ref, morph_df)
    test_df = intersect_ephys_morph(ephys_pred, morph_df)
    return ref_df, test_df


def intersect_ephys_morph(ephys_df, morph_df):
    morph_ids = morph_df.index.values

    # Get ephys data for cells with morphologies
    ids_with_morph_for_ephys = [s for s in morph_ids
                                if s in ephys_df.index.tolist()]
    ephys_df_joint = ephys_df.loc[ids_with_morph_for_ephys, :]

    # Only use morphs that have ephys
    mask = [s in ephys_df_joint.index.tolist() for s in morph_ids]
    morph_df_joint = morph_df.loc[mask, :]

    elmo_data = np.hstack([morph_df_joint.values,  ephys_df_joint.values])
    return pd.DataFrame(elmo_data, index=ephys_df_joint.index.values)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=RfPredictionParameters)
    main(**module.args)




