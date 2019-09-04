from builtins import zip
from builtins import range
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import logging
import json
import h5py
import os.path


def load_data(project="T301", use_noise=False, dendrite_type="all", need_structure=True, include_dend_type_null=False,
              limit_to_cortical_layers=None,
              params_file="/allen/programs/celltypes/workgroups/ivscc/nathang/single-cell-ephys/dev/default_spca_params.json",
              restriction_file=None,
              base_dir="/allen/programs/celltypes/workgroups/ivscc/nathang/single-cell-ephys/single_cell_ephys",
              step_num=50):
    # Import data
    metadata = pd.read_csv(os.path.join(base_dir, "fv_metadata_{:s}.csv".format(project)), index_col=0)

    specimen_ids = np.load(os.path.join(base_dir, "fv_ids_{:s}.npy".format(project)))
    step_subthresh = np.load(os.path.join(base_dir, "fv_step_subthresh_{:s}.npy".format(project)))
    subthresh_norm = np.load(os.path.join(base_dir, "fv_subthresh_norm_{:s}.npy".format(project)))
#     ramp_subthresh = np.load(os.path.join(base_dir, "fv_ramp_subthresh_{:s}.npy".format(project)))
    first_ap = np.load(os.path.join(base_dir, "fv_first_ap_{:s}.npy".format(project)))
    spiking = np.load(os.path.join(base_dir, "fv_spiking_{:s}.npy".format(project)))
    isi_shape = np.load(os.path.join(base_dir, "fv_isi_shape_{:s}.npy".format(project)))
    if use_noise:
        noise = np.load(os.path.join(base_dir, "fv_noise_{:s}.npy".format(project)))

    logging.info("Starting with {:d} cells".format(len(specimen_ids)))

    # Deal with weird yet-to-be-debugged values
#     bad_ids = [501566512, 593610936]
    bad_ids = [611520287]
    mask = np.array([i not in bad_ids for i in specimen_ids])
    orig_n_ids = len(specimen_ids)
    n_f = len(first_ap[0])
    lens = np.array([len(f) for f in first_ap])
    tmp_mask = lens != n_f
    if len(specimen_ids) == orig_n_ids:
        specimen_ids = specimen_ids[mask]
    if len(step_subthresh) == orig_n_ids:
        step_subthresh = step_subthresh[mask]
    if len(subthresh_norm) == orig_n_ids:
        subthresh_norm = subthresh_norm[mask]
#     if len(ramp_subthresh) == orig_n_ids:
#         ramp_subthresh = ramp_subthresh[mask]
    if len(first_ap) == orig_n_ids:
        first_ap = first_ap[mask]
    if len(spiking) == orig_n_ids:
        spiking = spiking[mask]
    if len(isi_shape) == orig_n_ids:
        isi_shape = isi_shape[mask]
    if use_noise and len(noise) == orig_n_ids:
        noise = noise[mask]

    if np.any(Series(specimen_ids).value_counts() > 1):
        logging.info("Handing duplicate specimen ids")
        mask = np.array([True] * len(specimen_ids))
        mask[np.flatnonzero(np.diff(specimen_ids) == 0)] = False

        specimen_ids = specimen_ids[mask]
        step_subthresh = step_subthresh[mask]
        subthresh_norm = subthresh_norm[mask]
#         ramp_subthresh = ramp_subthresh[mask]
        first_ap = first_ap[mask]
        spiking = spiking[mask]
        isi_shape = isi_shape[mask]
        if use_noise:
            noise = noise[mask]

    problem_ramp_ap = np.all(first_ap[:, 300:450] == 0, axis=1)
    logging.info("Cells with problem ramp AP: {:d}".format(int(np.sum(problem_ramp_ap))))

    specimen_ids = specimen_ids[~problem_ramp_ap]
    step_subthresh = step_subthresh[~problem_ramp_ap, :]
    subthresh_norm = subthresh_norm[~problem_ramp_ap, :]
#     ramp_subthresh = ramp_subthresh[~problem_ramp_ap, :]
    first_ap = first_ap[~problem_ramp_ap, :]
    spiking = spiking[~problem_ramp_ap, :]
    isi_shape = isi_shape[~problem_ramp_ap, :]

    if use_noise:
        logging.info("Using noise")
        noise = noise[~problem_ramp_ap]
        has_noise = np.array([arr is not None for arr in noise])
        noise = np.array(noise[has_noise].tolist())
        specimen_ids = specimen_ids[has_noise]
        step_subthresh = step_subthresh[has_noise, :]
        subthresh_norm = subthresh_norm[has_noise, :]
#         ramp_subthresh = ramp_subthresh[has_noise, :]
        first_ap = first_ap[has_noise, :]
        spiking = spiking[has_noise, :]
        isi_shape = isi_shape[has_noise, :]

        problem_noise = (np.all(noise[:, np.arange(0, 150) + 0 * 1410 + 60 + 7 * 150] == 0, axis=1) |
                         np.all(noise[:, np.arange(0, 150) + 1 * 1410 + 60 + 7 * 150] == 0, axis=1) |
                         np.all(noise[:, np.arange(0, 150) + 2 * 1410 + 60 + 7 * 150] == 0, axis=1) |
                         np.all(noise[:, np.arange(0, 150) + 3 * 1410 + 60 + 7 * 150] == 0, axis=1))

        specimen_ids = specimen_ids[~problem_noise]
        step_subthresh = step_subthresh[~problem_noise, :]
        subthresh_norm = subthresh_norm[~problem_noise, :]
#         ramp_subthresh = ramp_subthresh[~problem_noise, :]
        first_ap = first_ap[~problem_noise, :]
        spiking = spiking[~problem_noise, :]
        isi_shape = isi_shape[~problem_noise, :]
        noise = noise[~problem_noise, :]


    # Import metadata
    meta_df = metadata.set_index("specimen_id").loc[specimen_ids, :]
    meta_df.loc[meta_df["cre_reporter_status"].isnull(), "cre_reporter_status"] = "none"
    meta_df = merge_cre_lines(meta_df)
    meta_df["cre_w_status"] = "unlabeled"
    positive_ind = meta_df["cre_reporter_status"].str.endswith("positive")
    if positive_ind.any():
        meta_df.ix[positive_ind, "cre_w_status"] = meta_df.ix[positive_ind, "cre_line"]
    indeterminate_ind = meta_df["cre_reporter_status"].str.endswith("indeterminate")
    indeterminate_ind.fillna(False, inplace=True)
    if indeterminate_ind.any():
        meta_df.ix[indeterminate_ind, "cre_w_status"] = "indeterminate"
    struct_layer = {"1": "1", "2/3": "2/3", "4": "4", "5": "5", "6a": "6", "6b": "6"}
    meta_df["layer"] = "unk"
    for sl in struct_layer:
        meta_df.ix[[s.endswith(sl) if type(s) == str else False for s in meta_df["structure"]], "layer"] = struct_layer[sl]
    meta_df["cre_layer"] = meta_df["cre_w_status"] + " " + meta_df["layer"]
    meta_df["dendrite_type"] = [s.replace("dendrite type - ", "") if type(s) is str else np.nan for s in meta_df["dendrite_type"]]

    logging.info("Cells with Cre status indeterminate: {:d}".format(np.sum(meta_df["cre_w_status"] == "indeterminate")))
    logging.info("Cells with dendrite type NA: {:d}".format(np.sum(meta_df["dendrite_type"] == "NA")))
    logging.info("Cells with both indeterminate Cre status and dendrite type NA: {:d}".format(np.sum((meta_df["cre_w_status"] == "indeterminate") & (meta_df["dendrite_type"] == "NA"))))

    inclusion_mask = filter_by_dendrite_and_structure(meta_df, need_structure,
        dendrite_type, include_dend_type_null)

    if limit_to_cortical_layers is not None:
        inclusion_mask = inclusion_mask & meta_df["cortex_layer"].isin(limit_to_cortical_layers)
        logging.info("Cells in restricted cortical layers: {:d}".format(int(np.sum(inclusion_mask))))

    specimen_ids = specimen_ids[inclusion_mask]
    step_subthresh = step_subthresh[inclusion_mask, :]
    subthresh_norm = subthresh_norm[inclusion_mask, :]
#     ramp_subthresh = ramp_subthresh[inclusion_mask, :]
    first_ap = first_ap[inclusion_mask, :]
    spiking = spiking[inclusion_mask, :]
    isi_shape = isi_shape[inclusion_mask, :]
    if use_noise:
        noise = noise[inclusion_mask, :]
    meta_df = meta_df.loc[inclusion_mask, :]


    if restriction_file is not None:
        # Load file of IDs that the cells must be in
        restrict_ids = np.loadtxt(restriction_file)
        inclusion_mask = np.array([s in restrict_ids for s in specimen_ids])

        specimen_ids = specimen_ids[inclusion_mask]
        step_subthresh = step_subthresh[inclusion_mask, :]
        subthresh_norm = subthresh_norm[inclusion_mask, :]
#         ramp_subthresh = ramp_subthresh[inclusion_mask, :]
        first_ap = first_ap[inclusion_mask, :]
        spiking = spiking[inclusion_mask, :]
        isi_shape = isi_shape[inclusion_mask, :]
        if use_noise:
            noise = noise[inclusion_mask, :]
        meta_df = meta_df.loc[inclusion_mask, :]

    spca_zht_params, step_num = define_spca_parameters(filename=params_file)

    if "spiking_inst_freq" in spca_zht_params and "inst_freq_norm" in spca_zht_params:
        indices = spca_zht_params["spiking_inst_freq"][3]
        logging.debug("calculating inst_freq_norm with step_num {:d}".format(step_num))
        inst_freq_norm = spiking[:, indices]
        n_steps = len(indices) // step_num
        for i in range(n_steps):
            row_max = inst_freq_norm[:, i * step_num:(i + 1) * step_num].max(axis=1)
            row_max[row_max == 0] = 1.
            inst_freq_norm[:, i * step_num:(i + 1) * step_num] = inst_freq_norm[:, i * step_num:(i + 1) * step_num] / row_max[:, None]
    else:
        inst_freq_norm = None

    return specimen_ids, first_ap, isi_shape, step_subthresh, subthresh_norm, spiking, inst_freq_norm, meta_df


def filter_by_dendrite_and_structure(meta_df, need_structure, dendrite_type, include_dend_type_null):
    """Create mask for cells that pass metadata filters

    Parameters
    ----------
    metadata_df: DataFrame
        DataFrame of metadata
    need_structure: bool (optional, default False)
        Requires that structure is present (only used
        if metadata file is supplied)
    dendrite_type: str (optional, default 'all')
        Dendrite type for filtering ('all', 'spiny', 'aspiny') (only used
        if metadata file is supplied)
    include_dend_type_null: bool (optional, default True)
        Also include cells without a dendrite type available regardless of
        what `dendrite_type` is specified (only used
        if metadata file is supplied)

    Results
    -------
    inclusion_mask: array of shape (len(specimen_ids), )
        Boolean mask for filtered cells
    """
    # Refine the data set
    if need_structure:
        logging.info("Requiring structure and dendrite type; excluding dendrite type = NA")
        if dendrite_type == "all":
            inclusion_mask = np.array((meta_df["cre_w_status"] != "indeterminate") &
                              (~meta_df["structure"].isnull()) &
                              (~meta_df["dendrite_type"].isnull()) &
                              (meta_df["dendrite_type"] != "NA"))
        elif dendrite_type == "spiny":
            inclusion_mask = np.array((meta_df["cre_w_status"] != "indeterminate") &
                              (~meta_df["structure"].isnull()) &
                              (meta_df["dendrite_type"].isin(["spiny"])))
        elif dendrite_type == "aspiny":
            inclusion_mask = np.array((meta_df["cre_w_status"] != "indeterminate") &
                              (~meta_df["structure"].isnull()) &
                              (meta_df["dendrite_type"].isin(["aspiny", "sparsely spiny"])))
        else:
            raise ValueError("Not allowable value for dendrite type")
        logging.info("Cells with dendrite type and structure: {:d}".format(int(np.sum(inclusion_mask))))
    elif not include_dend_type_null:
        if dendrite_type == "all":
            inclusion_mask = np.array((meta_df["cre_w_status"] != "indeterminate") &
                              (~meta_df["dendrite_type"].isnull()) &
                              (meta_df["dendrite_type"] != "NA"))
        elif dendrite_type == "spiny":
            inclusion_mask = np.array((meta_df["cre_w_status"] != "indeterminate") &
                              (meta_df["dendrite_type"].isin(["spiny"])))
        elif dendrite_type == "aspiny":
            inclusion_mask = np.array((meta_df["cre_w_status"] != "indeterminate") &
                              (meta_df["dendrite_type"].isin(["aspiny", "sparsely spiny"])))
        else:
            raise ValueError("Not allowable value for dendrite type")
        logging.info("Requiring dendrite type; excluding dendrite type = NA")
        logging.info("Cells with dendrite type: {:d}".format(int(np.sum(inclusion_mask))))
    else:
        if dendrite_type == "all":
            inclusion_mask = np.array((meta_df["cre_w_status"] != "indeterminate") &
                              (meta_df["dendrite_type"].fillna("") != "NA"))
        elif dendrite_type == "spiny":
            inclusion_mask = np.array((meta_df["cre_w_status"] != "indeterminate") &
                              (meta_df["dendrite_type"].isin(["spiny"]) | meta_df["dendrite_type"].isnull()))
        elif dendrite_type == "aspiny":
            inclusion_mask = np.array((meta_df["cre_w_status"] != "indeterminate") &
                              (meta_df["dendrite_type"].isin(["aspiny", "sparsely spiny"]) | meta_df["dendrite_type"].isnull()))
        else:
            raise ValueError("Not allowable value for dendrite type")
        logging.info("Excluding dendrite type = NA")
        logging.info("Cells with dendrite type specified or missing (does not include NA): {:d}".format(int(np.sum(inclusion_mask))))

    return inclusion_mask


def load_organized_data(project, base_dir, params_file, dendrite_type,
                        use_noise=False, need_structure=False,
                        include_dend_type_null=True,
                        limit_to_cortical_layers=None):
    logging.info("in load_and_organize_data")
    (specimen_ids,
     first_ap,
     isi_shape,
     step_subthresh,
     subthresh_norm,
     spiking,
     inst_freq_norm,
     meta_df) = load_data(project=project,
                          base_dir=base_dir,
                          params_file=params_file,
                          use_noise=use_noise,
                          dendrite_type=dendrite_type,
                          need_structure=need_structure,
                          include_dend_type_null=include_dend_type_null,
                          limit_to_cortical_layers=limit_to_cortical_layers)

    data_for_spca = [
        {"data": first_ap,
         "part_keys": ["first_ap_v", "first_ap_dv"],
        },
        {"data": isi_shape,
         "part_keys": ["isi_shape"],
        },
        {"data": step_subthresh,
         "part_keys": ["step_subthresh"],
        },
        {"data": subthresh_norm,
         "part_keys": ["subthresh_norm"],
        },
        {"data": spiking,
         "part_keys": ["spiking_rate", "spiking_inst_freq", "spiking_updown", "spiking_peak_v",
                       "spiking_fast_trough_v",
                       "spiking_threshold_v", "spiking_width"],
        },
    ]
    if use_noise:
        data_for_spca.append(
            {"data": noise,
             "part_keys": ["noise_rate", "noise_inst_freq", "noise_updown", "noise_peak_v",
                           "noise_fast_trough_v",
                           "noise_threshold_v", "noise_width"],
            },
        )

    if inst_freq_norm is not None:
        data_for_spca.append({
            "data": inst_freq_norm,
            "part_keys": ["inst_freq_norm"],
        })

    return data_for_spca, specimen_ids


def load_h5_data(h5_fv_file, params_file, metadata_file=None, dendrite_type="all",
        need_structure=False,
        include_dend_type_null=True,
        limit_to_cortical_layers=None,
        id_file=None):
    """Load dictionary for sPCA processing from HDF5 file with specified
       metadata filters

    Parameters
    ----------
    h5_fv_file: str
        Path to feature vector HDF5 file
    params_file: str
        Path to sPCA parameters JSON file
    metadata_file: str (optional, default None)
        Path to metadata CSV file
    dendrite_type: str (optional, default 'all')
        Dendrite type for filtering ('all', 'spiny', 'aspiny') (only used
        if metadata file is supplied)
    need_structure: bool (optional, default False)
        Requires that structure is present (only used
        if metadata file is supplied)
    include_dend_type_null: bool (optional, default True)
        Also include cells without a dendrite type available regardless of
        what `dendrite_type` is specified (only used
        if metadata file is supplied)
    limit_to_cortical_layers: list (optional, default None)
        List of cortical layers that metadata must match for inclusion (only used
        if metadata file is supplied)
    id_file: str (optional, default None)
        Path to text file with IDs to use

    Results
    -------
    data_for_spca: dict
        Dictionary of data sets for sPCA analysis
    specimen_ids: array
        The specimen IDs for the cells in the data sets
    """

    f = h5py.File(h5_fv_file, "r")
    spca_zht_params, step_num = define_spca_parameters(filename=params_file)

    specimen_ids = f["ids"][...]
    logging.info("Starting with {:d} cells".format(len(specimen_ids)))

    # Identify cells with no ramp spike
    first_ap_v = f["first_ap_v"][...]

    # Expected to have three equal-length AP waveforms
    n_bins = first_ap_v.shape[1] // 3

    # Ramp waveform expected to be last
    ramp_mask = ~np.all(first_ap_v[:, -n_bins:] == 0, axis=1)
    logging.info("{} cells have no ramp AP".format(np.sum(ramp_mask == False)))

    if metadata_file is not None:
        logging.debug("Using metadata file {}".format(metadata_file))
        metadata = pd.read_csv(metadata_file, index_col=0)
        mask = mask_for_metadata(specimen_ids, metadata,
            dendrite_type, need_structure,
            include_dend_type_null, limit_to_cortical_layers)
        mask = mask & ramp_mask
    else:
        mask = ramp_mask

    if id_file is not None:
        with open(id_file, "r") as id_f:
            include_id_list = [int(line.strip("\n")) for line in id_f]
        id_mask = np.array([spec_id in include_id_list for spec_id in specimen_ids])
        mask = mask & id_mask

    data_for_spca = {}
    for k in spca_zht_params:
        if k not in f.keys():
            logging.debug("{} not found in HDF5 file".format(k))
            continue
        data = f[k][mask, :]
        data_for_spca[k] = data

    # Calculate additional data set if requested
    if ("inst_freq" in spca_zht_params and "inst_freq_norm" in spca_zht_params
        and "inst_freq" in f.keys()):
        logging.debug("inst_freq_norm will be calculated from inst_freq")
        indices = spca_zht_params["inst_freq"][3]
        logging.debug("calculating inst_freq_norm with step_num {:d}".format(step_num))
        inst_freq_data = f["inst_freq"][...]
        if indices is not None:
            inst_freq_norm = inst_freq_data[mask, :][:, indices]
        else:
            inst_freq_norm = inst_freq_data[mask, :]
        n_steps = len(indices) // step_num
        for i in range(n_steps):
            row_max = inst_freq_norm[:, i * step_num:(i + 1) * step_num].max(axis=1)
            row_max[row_max == 0] = 1. # handle divide-by-zero issues
            inst_freq_norm[:, i * step_num:(i + 1) * step_num] = inst_freq_norm[:, i * step_num:(i + 1) * step_num] / row_max[:, None]
        data_for_spca["inst_freq_norm"] = inst_freq_norm
    f.close()

    specimen_ids = specimen_ids[ramp_mask & mask]
    logging.info("Loaded data for {} cells".format(len(specimen_ids)))


    return data_for_spca, specimen_ids


def mask_for_metadata(specimen_ids, metadata_df, dendrite_type="all",
        need_structure=False, include_dend_type_null=True,
        limit_to_cortical_layers=None):
    """Create mask for cells that pass metadata filters

    Parameters
    ----------
    specimen_ids: array
        Specimen IDs for cells to filter
    metadata_df: DataFrame
        DataFrame of metadata
    dendrite_type: str (optional, default 'all')
        Dendrite type for filtering ('all', 'spiny', 'aspiny') (only used
        if metadata file is supplied)
    need_structure: bool (optional, default False)
        Requires that structure is present (only used
        if metadata file is supplied)
    include_dend_type_null: bool (optional, default True)
        Also include cells without a dendrite type available regardless of
        what `dendrite_type` is specified (only used
        if metadata file is supplied)
    limit_to_cortical_layers: list (optional, default None)
        List of cortical layers that metadata must match for inclusion (only used
        if metadata file is supplied)

    Results
    -------
    mask: array of shape (len(specimen_ids), )
        Boolean mask for filtered cells
    """

    # Limit to specimen_ids
    meta_df = metadata_df.set_index("specimen_id").loc[specimen_ids, :]

    # Reformat metadata information
    meta_df.loc[meta_df["cre_reporter_status"].isnull(), "cre_reporter_status"] = "none"
    meta_df = merge_cre_lines(meta_df)
    meta_df["cre_w_status"] = "unlabeled"
    positive_ind = meta_df["cre_reporter_status"].str.endswith("positive")
    if positive_ind.any():
        meta_df.ix[positive_ind, "cre_w_status"] = meta_df.ix[positive_ind, "cre_line"]
    indeterminate_ind = meta_df["cre_reporter_status"].str.endswith("indeterminate")
    indeterminate_ind.fillna(False, inplace=True)
    if indeterminate_ind.any():
        meta_df.ix[indeterminate_ind, "cre_w_status"] = "indeterminate"

    struct_layer = {"1": "1", "2/3": "2/3", "4": "4", "5": "5", "6a": "6", "6b": "6"}
    meta_df["layer"] = "unk"
    for sl in struct_layer:
        meta_df.ix[[s.endswith(sl) if type(s) == str else False for s in meta_df["structure"]], "layer"] = struct_layer[sl]
    meta_df["cre_layer"] = meta_df["cre_w_status"] + " " + meta_df["layer"]
    meta_df["dendrite_type"] = [s.replace("dendrite type - ", "") if type(s) is str else np.nan for s in meta_df["dendrite_type"]]

    logging.info("Cells with Cre status indeterminate: {:d}".format(np.sum(meta_df["cre_w_status"] == "indeterminate")))
    logging.info("Cells with dendrite type NA: {:d}".format(np.sum(meta_df["dendrite_type"] == "NA")))
    logging.info("Cells with both indeterminate Cre status and dendrite type NA: {:d}".format(np.sum((meta_df["cre_w_status"] == "indeterminate") & (meta_df["dendrite_type"] == "NA"))))


    dend_struct_mask = filter_by_dendrite_and_structure(meta_df, need_structure,
        dendrite_type, include_dend_type_null)

    if limit_to_cortical_layers is not None:
        layer_mask = meta_df["cortex_layer"].isin(limit_to_cortical_layers)
        logging.info("Cells in restricted cortical layers: {:d}".format(int(np.sum(inclusion_mask))))
    else:
        layer_mask = np.ones_like(dend_struct_mask, dtype=bool)
    return dend_struct_mask & layer_mask


def define_spca_parameters(filename="/allen/programs/celltypes/workgroups/ivscc/nathang/single-cell-ephys/dev/default_spca_params.json"):
    # Parameters found
    # (n_components, n nonzero component list, use_corr, data_range)
    with open(filename, "r") as f:
        json_data = json.load(f)

    spca_zht_params = {}
    for k in json_data:
        d = json_data[k]

        if d["range"] is None:
            indices = None
        elif len(d["range"]) == 2:
            indices = np.arange(d["range"][0], d["range"][1])
        else:
            range_list = []
            for a, b in zip(d["range"][:-1:2], d["range"][1::2]):
                range_list.append(np.arange(a, b))
            indices = np.hstack(range_list)
        spca_zht_params[k] = (
            d["n_components"],
            d["nonzero_component_list"],
            d["use_corr"],
            indices,
        )


    if "inst_freq_norm" in json_data and "step_num" in json_data["inst_freq_norm"]:
        step_num = json_data["inst_freq_norm"]["step_num"]
    else:
        step_num = 50 # default value

    return spca_zht_params, step_num


def load_data_with_ids(id_list, project="T301", use_noise=False, dendrite_type="all"):
    (specimen_ids, first_ap, isi_shape, step_subthresh, subthresh_norm,
    spiking, inst_freq_norm, meta_df) = load_data(project=project,
                                                  use_noise=use_noise,
                                                  dendrite_type=dendrite_type,
                                                  need_structure=False,
                                                  include_dend_type_null=True)

    inclusion_mask = np.array([s in id_list for s in specimen_ids])

    specimen_ids = specimen_ids[inclusion_mask]
    step_subthresh = step_subthresh[inclusion_mask, :]
    subthresh_norm = subthresh_norm[inclusion_mask, :]
    first_ap = first_ap[inclusion_mask, :]
    spiking = spiking[inclusion_mask, :]
    isi_shape = isi_shape[inclusion_mask, :]
    meta_df = meta_df.loc[inclusion_mask, :]
    inst_freq_norm = inst_freq_norm[inclusion_mask, :]

    return specimen_ids, first_ap, isi_shape, step_subthresh, subthresh_norm, spiking, inst_freq_norm, meta_df


def merge_cre_lines(df):
    new_df = df.copy()
    lines_to_merge = {
        "Ntsr1-Cre": "Ntsr1-Cre_GN220",
        "Pvalb-IRES-Cre;Pvalb-IRES-Cre": "Pvalb-IRES-Cre",
        "Vip-IRES-Cre;Vip-IRES-Cre": "Vip-IRES-Cre",
        "Vipr2-IRES2-Cre;Vipr2-IRES2-Cre": "Vipr2-IRES2-Cre",
        "Chat-IRES-Cre-neo;Chat-IRES-Cre-neo": "Chat-IRES-Cre-neo",
        "Sst-IRES-FlpO;Nos1-CreERT2": "Nos1-CreERT2;Sst-IRES-FlpO",
    }

    new_df["cre_line"] = [lines_to_merge[c] if c in lines_to_merge else c
                          for c in df["cre_line"]]
    return new_df
