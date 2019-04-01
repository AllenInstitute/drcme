import numpy as np
from . import spca_zht as szht
from . import load_data as ld
import logging


def consolidate_spca(spca_results, data_for_spca, spca_zht_params, pev_threshold=0.01):
    Z_list = []
    component_record = []
    for ds in data_for_spca:
        data = ds["data"]
        for k in ds["part_keys"]:
            indices = spca_zht_params[k][3]
            logging.debug("{:s} pev total: {:g}".format(k, np.sum(spca_results[k]["pev"])))
            above_thresh = spca_results[k]["pev"] >= pev_threshold
            d = data[:, indices]
            Z = d.dot(spca_results[k]["loadings"][:, above_thresh])
            Z_list.append(Z)
            component_record.append({"key": k, "indices": np.flatnonzero(above_thresh).tolist()})

    combo_orig = np.hstack(Z_list)

    combo = (combo_orig - combo_orig.mean(axis=0)) / combo_orig.std(axis=0)
    logging.debug("Total sPCA dimensions: ({:d}, {:d})".format(*combo.shape))
    return combo, component_record


def spca_on_all_data(data_for_spca, spca_zht_params):
    spca_results = {}

    for ds in data_for_spca:
        data = ds["data"]
        for k in ds["part_keys"]:
            logging.info("Key " + k)
            logging.info("Processing " + k + " ... ")
            n_components, para, use_corr, indices = spca_zht_params[k]
            d = data[:, indices]
            spca_fit = szht.spca_zht(d, K=n_components, sparse="varnum", para=para,
                                eps_conv=1.5e-3, use_corr=use_corr, trace=True)
            spca_results[k] = spca_fit.copy()
            logging.info("pev = " + str(spca_fit["pev"]))
            logging.info("done")
    return spca_results
