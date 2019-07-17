from builtins import str
import numpy as np
from . import spca_zht as szht
from . import load_data as ld
from sklearn.decomposition import PCA
import logging


def consolidate_spca(spca_results, adj_exp_var_threshold=0.01):
    """Combine and z-score individual data set sPCs into single matrix

    Parameters
    ----------
    spca_results: dict
        Dictionary of loadings, adjusted explained variances, and transformed values
    adj_exp_var_threshold: float (default 0.01)
        Minimum adjusted explained variance level to retain sPCs

    Returns:
        combined_matrix: array (n_samples by n total sPCs)
            Z-scored matrix of all sPC-transformed values
        component_record: list
            List of kept component information for each key of spca_results
    """
    Z_list = []
    component_record = []
    for k in spca_results:
        r = spca_results[k]
        above_thresh = r["pev"] >= adj_exp_var_threshold
        Z_list.append(r["transformed_values"][:, above_thresh])
        component_record.append({"key": k, "indices": np.flatnonzero(above_thresh).tolist()})

    combo_orig = np.hstack(Z_list)
    combo = (combo_orig - combo_orig.mean(axis=0)) / combo_orig.std(axis=0)
    logging.debug("Total sPCA dimensions: ({:d}, {:d})".format(*combo.shape))
    return combo, component_record


def select_data_subset(data_for_spca, spca_params):
    """Select data sets and indices defined by `spca_params`

    Parameters
    ----------
    data_for_spca: dict
        Dictionary of feature vectors
    spca_params: dict
        Dictionary of sPCA analysis parameters

    Returns
    -------
    Dictionary of data subsets defined by sPCA parameters
    """

    subset_for_spca = {}
    for k in spca_params:
        if k not in data_for_spca:
            raise ValueError("requested key {} not found in data dictionary".format(k))

        data = data_for_spca[k]
        _, _, _, indices = spca_params[k]
        if indices is None:
            subset_for_spca[k] = data
        else:
            subset_for_spca[k] = data[:, indices]

    return subset_for_spca


def spca_on_all_data(data_for_spca, spca_params, max_iter=200, eps_conv=1.5e-3):
    """Compute sPCA for data with specified parameters

    Parameters
    ----------
    data_for_spca: dict
        Dictionary of feature vectors
    spca_params: dict
        Dictionary of sPCA analysis parameters
    max_iter: int (default 200)
        Maximum sPCA iterations
    eps_conv: float (default 1.5e-3)
        Convergence criterion

    Returns
    -------
    Dictionary of sPCA results with keys:
        "loadings": sPC loadings (n_features by n_components array)
        "pev": adjusted explained variance (n_components-length array)
        "transformed values": sPC-transformed values (n_samples by n_components array)
    """
    spca_results = {}

    for k in spca_params:
        if k not in data_for_spca:
            raise ValueError("requested key {} not found in data dictionary".format(k))

        data = data_for_spca[k]
        logging.info("Processing " + k + " ... ")
        n_components, para, use_corr, _ = spca_params[k]
        if len(para) == 0: # special case - do regular PCA
            logging.info("Using regular PCA")
            pca = PCA(n_components=n_components)
            transformed_values = pca.fit_transform(data)
            fit = {
                "loadings": pca.components_.T,
                "pev": pca.explained_variance_ratio_,
                "transformed_values": transformed_values,
            }
            spca_results[k] = fit.copy()
        else:
            fit = szht.spca_zht(data, K=n_components, sparse="varnum", para=para,
                                eps_conv=eps_conv, max_iter=max_iter,
                                use_corr=use_corr, trace=True)
            spca_results[k] = fit.copy()
            spca_results[k]["transformed_values"] = data.dot(fit["loadings"])
        logging.info(
            "adjusted explained variance = {:s}; total = {:.2g}".format(
                str(fit["pev"]), fit["pev"].sum())
        )
    return spca_results
