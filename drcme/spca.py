"""
The :mod:`drcme.spca` module contains functions for performing
sparse principal component analysis.
"""

import numpy as np
import scipy
import scipy.linalg as sl
from numba import jit, njit
from . import load_data as ld
from .delcol import delcol
from sklearn.decomposition import PCA
import logging


def consolidate_spca(spca_results, adj_exp_var_threshold=0.01):
    """Combine and z-score individual data set sPCs into single matrix

    Parameters
    ----------
    spca_results : dict
        Dictionary of loadings, adjusted explained variances, and
        transformed values
    adj_exp_var_threshold : float, optional
        Minimum adjusted explained variance level (default 0.01) to
        retain sPCs

    Returns :
    combined_matrix : (n_samples, n total sPCs) array
        Z-scored matrix of all sPC-transformed values
    component_record : list
        List of kept component information for each key of
        `spca_results`
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
    data_for_spca : dict
        Dictionary of feature vectors
    spca_params : dict
        Dictionary of sPCA analysis parameters

    Returns
    -------
    dict
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


def spca_on_all_data(data_for_spca, spca_params, max_iter=500, eps_conv=1.5e-3):
    """Compute sPCA for multiple data sets with specified parameters

    Parameters
    ----------
    data_for_spca : dict
        Dictionary of feature vectors
    spca_params : dict
        Dictionary of sPCA analysis parameters. Must have all the keys
        found in `data_for_spca`
    max_iter : int, optional
        Maximum number of sPCA iterations (default 500)
    eps_conv : float, optional
        Convergence criterion

    Returns
    -------
    dict
        Dictionary of dictionaries each containing sPCA results with
        keys: "loadings": sPC loadings (n_features by n_components
        array), "pev": adjusted explained variance (n_components-length
        array), "transformed values": sPC-transformed values (n_samples
        by n_components array)
    """
    spca_results = {}

    for k in spca_params:
        if k not in data_for_spca:
            raise ValueError("requested key {} not found in data dictionary".format(k))

        data = data_for_spca[k]
        logging.info("Processing " + k + " ... ")
        n_components, para, use_corr, _ = spca_params[k]

        spca_results[k] = spca_on_data_set(
            data, n_components, para, use_corr, max_iter, eps_conv)

    return spca_results


def spca_on_data_set(data, n_components, para, use_corr, max_iter=500, eps_conv=1.5e-3):
    """ Compute sPCA for a single data set

    This is essentially a wrapper function with a simplified interface
    for :func:`spca_zht` that embeds the typical
    expectations/conventions from the sPCA configuration files.

    Parameters
    ----------
    data : array
        A sample-by-feature data matrix
    n_components : int
        Number of components
    para : (K, ) array
        Parameter array of the number of non-zero sparse loadings that
        will be obtained by the algorithm. If K == 0, performs standard
        PCA.
    use_corr : bool
        Whether to scale the data by the standard deviation
    max_iter : int, optional
        Maximum number of sPCA iterations (default 500)
    eps_conv : float, optional
        Convergence criterion

    Returns
    -------
    dict
        Contains the keys: "loadings": sPC loadings (n_features by
        n_components array), "pev": adjusted explained variance
        (n_components-length array), "transformed values":
        sPC-transformed values (n_samples by n_components array)
    """
    if len(para) == 0: # special case - do regular PCA
        logging.info("Using regular PCA")
        pca = PCA(n_components=n_components)
        transformed_values = pca.fit_transform(data)
        fit = {
            "loadings": pca.components_.T,
            "pev": pca.explained_variance_ratio_,
            "transformed_values": transformed_values,
        }
    else:
        fit = spca_zht(data, K=n_components, sparse="varnum", para=para,
                            eps_conv=eps_conv, max_iter=max_iter,
                            use_corr=use_corr, trace=True)
        fit["transformed_values"] = data.dot(fit["loadings"])

    logging.info(
        "adjusted explained variance = {:s}; total = {:.2g}".format(
            str(fit["pev"]), fit["pev"].sum())
    )

    return fit


def orig_mean_and_std_for_zscore(spca_results, orig_data, spca_params,
        pev_threshold=0.01):
    """Recover mean and standard deviation of z-scored sPCs

    Parameters
    ----------
    spca_results : dict
        Output of :func:`spca_on_all_data`
    orig_data : dict
        Dictionary of data sets
    spca_params : dict
        Dictionary of sPCA parameters
    pev_threshold : float, optional
        Minimum adjusted explained variance level (default 0.01) to
        retain sPCs

    Returns
    -------
    mean : array
        Mean values of sPCs
    std : array
        Standard deviations of sPCs
    """
    Z_list = []
    subset_data = select_data_subset(orig_data, spca_params)
    for k, d in subset_data.items():
        above_thresh = spca_results[k]["pev"] >= pev_threshold
        Z = d.dot(spca_results[k]["loadings"][:, above_thresh])
        if np.any(np.isnan(Z)):
            print("NaNs found", k)
        Z_list.append(Z)

    combo_orig = np.hstack(Z_list)
    return combo_orig.mean(axis=0), combo_orig.std(axis=0)


def spca_transform_new_data(spca_results, new_data, spca_zht_params, orig_mean, orig_std, pev_threshold=0.01):
    """Transform and z-score new data with existing loadings

    Results from new data are z-scored with original mean and standard
    deviation so that they are directly comparable to values in
    `spca_results`.

    Parameters
    ----------
    spca_results : dict
        Output of :func:`spca_on_all_data` with original loadings
        to be applied to `new_data`
    new_data : dict
        Dictionary of data sets for transformation
    orig_mean : array
        Mean values of original sPCs before z-scoring
    orig_std : array
        Standard deviations of sPCs before z-scoring
    pev_threshold : float, optional
        Minimum adjusted explained variance level (default 0.01) to
        retain sPCs

    Returns
    -------
    array
        Transformed and z-scored sPC values
    """
    Z_list = []
    subset_data = select_data_subset(new_data, spca_zht_params)
    for k, d in subset_data.items():
        above_thresh = spca_results[k]["pev"] >= pev_threshold
        Z = d.dot(spca_results[k]["loadings"][:, above_thresh])
        if np.any(np.isnan(Z)):
            print("NaNs found", k)
        Z_list.append(Z)

    combo_new = np.hstack(Z_list)
    combo = (combo_new - orig_mean) / orig_std
    return combo


def spca_zht(data, K, para, type="predictor", sparse="penalty", use_corr=False,
             lambda_val=1e-6, max_iter=200, trace=False, eps_conv=1e-3):
    """ Perform sparse principal component analysis

    This is a Python port of the sparse PCA algorithm of the R
    elasticnet package
    (https://cran.r-project.org/web/packages/elasticnet/index.html),
    based on Zou, Hastie, and Tibshirani (2006).

    Parameters
    ----------
    data : array
        A sample-by-feature data matrix (if `type` is "predictor"), or a
        sample covariance/correlation matrix (if `type` is "Gram")
    K : int
        Number of components
    para : (K, ) array
        Parameter array. If `sparse` is "penalty", `para` should contain
        the L1 norm penalty parameters. If `sparse` is "varnum", `para`
        should contain the number of non-zero sparse loadings that will
        be obtained by the algorithm.
    type : {'predictor', 'Gram'}, optional
        If "predictor" (default), `data` is a data matrix. If "Gram",
        `data` is a covariance/correlation matrix.
    sparse : {'penalty', 'varnum'}, optional
        If "penalty" (default), `para` contains the L1 norm penalties.
        If "varnum", `para` contains the desired number of non-zero
        loadings.
    lambda : float, optional
        Quadratic penalty parameter
    use_corr : bool, optional
        Whether to scale the data by the standard deviation
    max_iter : int, optional
        Maximum number of iterations
    trace : bool
        Whether to log progress at level INFO
    eps_conv : float
        Convergence criterion

    Returns
    -------
    dict
        Contains the sparse PCA loadings ("loadings"), adjusted
        percentage of explained variance ("pev") and total variance of
        all the predictors ("var_all")
    """
    x = data.copy().astype(np.float64)
    if type == "predictor":
        n = x.shape[0]
        p = x.shape[1]

        if float(n) / p >= 100:
            logging.info("You may wish to restart and use a more efficient way - " +
                   "let the argument x be the sample covariance/correlation " +
                   "matrix and set type=Gram")
        if trace:
            if use_corr:
                logging.info("Centering and scaling the data")
            else:
                logging.info("Centering the data")
        x = _scale(x, scale=use_corr)
    elif type == "Gram":
        x = _rootmatrix(x)

    u, d, v = np.linalg.svd(x, full_matrices=False)
    totalvariance = (d ** 2).sum()
    alpha = v.T[:, :K]
    beta = alpha.copy()
    if trace:
        logging.info("Calculating initial beta")
    for i in range(K):
        y = np.squeeze(x.dot(alpha[:, i]))
        beta[:, i] = _solvebeta(x, y, paras=(lambda_val, para[i]), sparse=sparse)
    xtx = x.T.dot(x)
    temp = beta.copy()
    normtemp = np.sqrt((temp ** 2).sum(axis=0))
    normtemp[normtemp == 0] = 1
    temp = temp / normtemp
    k = 0
    diff = 1
    if trace:
        logging.info("Refining loadings")
    while k < max_iter and diff > eps_conv:
        alpha = xtx.dot(beta)
        z = np.linalg.svd(alpha, full_matrices=False)
        alpha = z[0].dot(z[2])
        for i in range(K):
            y = np.squeeze(x.dot(alpha[:, i]))
            beta[:, i] = _solvebeta(x, y, paras=(lambda_val, para[i]), sparse=sparse)
        normbeta = np.sqrt((beta ** 2).sum(axis=0))
        normbeta[normbeta == 0] = 1
        beta2 = beta / normbeta
        diff = np.max(np.abs(beta2 - temp))
        temp = beta2.copy()
        if trace:
            if k % 10 == 0:
                logging.info("Iterations {:d}: difference on this step was {:0.5f}".format(k, diff))
        k += 1
    normbeta = np.sqrt((beta ** 2).sum(axis=0))
    normbeta[normbeta == 0] = 1
    beta = beta / normbeta
    u = x.dot(beta)
    q, R = np.linalg.qr(u)
    pev = np.diag(R ** 2) / totalvariance
    return {
        "loadings": beta,
        "pev": pev,
        "var_all": totalvariance,
    }


def _scale(data, scale=True):
    ''' Centers columns and optionally divides by the (centered) std'''

    x = data - np.mean(data, axis=0)
    if scale:
        x = x / x.std(axis=0)

    return x


@njit
def _rootmatrix(x):
    d, v = np.linalg.eig(x)
    d = (d + np.abs(d)) / 2.
    return v.dot(np.diag(np.sqrt(d)).dot(v.T))


def _solvebeta(x, y, paras, max_steps=None, sparse=None, eps=2.22e-16):
    if not sparse:
        sparse = "penalty"

    nm = x.shape
    n = nm[0]
    m = nm[1]
    im = np.arange(m)

    lambda_val = paras[0]
    if lambda_val > 0:
        max_vars = m
    elif lambda_val == 0:
        max_vars = min(m, n - 1)
        if (m == n):
            max_vars = m

    d1 = np.sqrt(lambda_val)
    d2 = 1. / np.sqrt(1. + lambda_val)
    Cvec = y.dot(x) * d2
    ssy = (y ** 2).sum()
    residuals = np.hstack((y, np.zeros(m)))
    if max_steps is None:
        max_steps = 50 * max_vars
    penalty = np.array([np.max(np.abs(Cvec))])
    dropid = None

    if (sparse == "penalty") and (penalty * 2. / d2 <= paras[1]):
        beta = np.zeros(m)
    else:
        beta = np.zeros(m)
        first_in = np.zeros(m, dtype=np.int32)
        active = np.array([], dtype=np.int32)
        ignores = np.array([], dtype=np.int32)
        drops = np.array([False])
        Sign = np.array([])
        R_data = None
        R_rank = None
        k = 0
        while (k < max_steps - 1) and (len(active) < max_vars - len(ignores)):
            action = np.array([])
            if k == 0:
                inactive = im.copy()
            else:
                mask = np.ones(im.shape, dtype=np.bool_)
                mask[np.concatenate((active, ignores))] = False
                inactive = im[mask]
            C = Cvec[inactive]
            Cmax = np.max(np.abs(C))
            if not np.any(drops):
                new_mask = (np.abs(C) >= Cmax)
                C = C[~new_mask]
                new = inactive[new_mask]
                for inew in new:
                    sign_list = []
                    ignores_list = []
                    active_list = []
                    action_list = []
                    if len(active) > 0:
                        xold = x[:, active]
                    else:
                        xold = np.array([])
                    R = _updateRR(x[:, inew], (R_data, R_rank), xold, lambda_val)
                    R_data = R[0]
                    R_rank = R[1]
                    if R_rank == len(active):
                        nR = np.arange(len(active))
                        R_data = R_data[nR, :][:, nR]
                        R_rank = len(active)
                        ignores_list.append(inew)
                        action_list.append(-inew)
                    else:
                        if first_in[inew] == 0:
                            first_in[inew] = k
                        active_list.append(inew)
                        sign_list.append(np.sign(Cvec[inew]))
                        action_list.append(inew)
                    ignores = np.concatenate((ignores, np.array(ignores_list, dtype=np.int32)))
                    active = np.concatenate((active, np.array(active_list, dtype=np.int32)))
                    action = np.concatenate((action, np.array(action_list, dtype=np.int32)))
                    Sign = np.concatenate((Sign, np.array(sign_list, dtype=np.int32)))
            else:
                action = -dropid

            Gi1 = sl.solve_triangular(R_data,
                    sl.solve_triangular(R_data, Sign, trans=1))
            beta2 = beta.copy()
            drops, gamhat, A, w, residuals, Cvec = determine_drops(x, beta, residuals, Gi1, Sign, d1, d2, active, ignores,
                lambda_val, n, m, C, Cmax, eps)
            beta[active] = beta[active] + gamhat * w
            penalty = np.append(penalty, penalty[k] - np.abs(gamhat * A))
            if (sparse == "penalty") and (penalty[-1] * 2. / d2 <= paras[1]):
                s1 = penalty[-1] * 2. / d2
                s2 = penalty[-2] * 2. / d2
                beta = (s2 - paras[1]) / (s2 - s1) * beta + (paras[1] - s1) / (s2 - s1) * beta2
                beta *= d2
                break
            if np.any(drops):
                dropid = np.arange(len(drops))[drops]
                for id in dropid[::-1]:
                    R_data, R_rank = _downdateR((R_data, R_rank), id + 1)
                dropid = active[drops]
                beta[dropid] = 0.
                active = active[~drops]
                Sign = Sign[~drops]
            if (sparse == "varnum") and (len(active) >= paras[1]):
                break
            k += 1

    return beta

@njit
def determine_drops(x, beta, residuals, Gi1, Sign, d1, d2, active, ignores, lambda_val,
        n, m, C, Cmax, eps):
    A = 1. / np.sqrt((Gi1 * Sign).sum())
    w = A * Gi1
    u1 = (x[:, active].dot(w * d2))
    u2 = np.zeros(m)
    u2[active] = d1 * d2 * w
    u = np.concatenate((u1, u2))
    if lambda_val > 0:
        max_vars = m - len(ignores)
    elif lambda_val == 0:
        max_vars = min(m - len(ignores), n - 1)
    if len(active) == max_vars - len(ignores):
        gamhat = Cmax / A
    else:
        mask = np.ones(x.shape[1], dtype=np.bool_)
        mask[np.concatenate((active, ignores))] = False
        a = (u1.dot(x[:, mask] + d1 * u2[mask])) * d2
        gam = np.concatenate(((Cmax - C)/(A - a), (Cmax + C) / (A + a)))
        if np.sum(gam > eps) > 0:
            gamhat = min(gam[gam > eps].min(), Cmax / A)
        else:
            gamhat = Cmax / A
    dropid = None
    b1 = beta[active]
    z1 = -b1 / w
    z1_for_min = z1[z1 > eps]
    if len(z1_for_min) > 0:
        zmin = min(z1_for_min.min(), gamhat)
    else:
        zmin = gamhat
    if zmin < gamhat:
        gamhat = zmin
        drops = z1 == zmin
    else:
        drops = np.array([False])
    residuals = residuals - (gamhat * u)
    Cvec = (residuals[:n].T.dot(x) + d1 * residuals[n:]) * d2

    return drops, gamhat, A, w, residuals, Cvec


def _updateRR(xnew, R, xold, lambda_val, eps=2.22e-16):
    xtx = ((xnew ** 2).sum() + lambda_val) / (1. + lambda_val)
    norm_xnew = np.sqrt(xtx)
    if R[0] is None:
        R = (np.array(norm_xnew).reshape(1, -1), 1)
        return R

    R_data, R_rank = R
    Xtx = xnew.dot(xold) / (1. + lambda_val)
    r = sl.solve_triangular(R_data, Xtx, trans=1)
    rpp = norm_xnew ** 2 - (r ** 2).sum()
    rank = R_rank
    if rpp <= eps:
        rpp = eps
    else:
        rpp = np.sqrt(rpp)
        rank = rank + 1
    new_R_data = np.zeros((R_data.shape[0] + 1, R_data.shape[1] + 1))
    new_R_data[:R_data.shape[0], :R_data.shape[1]] = R_data
    new_R_data[:R_data.shape[1], -1] = r
    new_R_data[-1, -1] = rpp
    R = (new_R_data, rank)
    return R


def _downdateR(R, k=None):
    p = R[0].shape[0]
    if p == 1:
        return None

    if not k:
        k = p
    R = (_delcol(R[0], np.ones(p), k)[:-1, :], p - 1)
    return R


def _delcol(r, z, k=None):
    p = r.shape[0]
    if not k:
        k = p
    mask = np.array([True] *  r.shape[1])
    mask[k - 1] = False
    r = r[:, mask]
    z = np.reshape(z, newshape=(-1, 1))
    dz = z.shape
    r, z = delcol(r, k, z)
    return r