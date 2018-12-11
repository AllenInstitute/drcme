#!/usr/bin/python

import numpy as np
import scipy
import lars_fort
import logging


def spca_zht(data, K, para, type="predictor", sparse="penalty", use_corr=False,
             lambda_val=1e-6, max_iter=200, trace=False, eps_conv=1e-3):
    x = data.copy()
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

    x = data - data.mean(axis=0)
    if scale:
        x = x / x.std(axis=0)

    return x


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
        max_vars = np.min(m, n - 1)
        if (m == n):
            max_vars = m

    d1 = np.sqrt(lambda_val)
    d2 = 1. / np.sqrt(1. + lambda_val)
    Cvec = y.dot(x) * d2
    ssy = (y ** 2).sum()
    residuals = np.append(y, np.zeros(m))
    if not max_steps:
        max_steps = 50 * max_vars
    penalty = np.array([np.max(np.abs(Cvec))])
    if (sparse == "penalty") and (penalty * 2. / d2 <= paras[1]):
        beta = np.zeros(m)
    else:
        beta = np.zeros(m)
        first_in = np.zeros(m, dtype=int)
        active = np.array([], dtype=int)
        ignores = np.array([], dtype=int)
        actions = range(max_steps)
        drops = False
        Sign = np.array([])
        R = None
        k = 0
        while (k < max_steps - 1) and (len(active) < max_vars - len(ignores)):
            action = np.array([])
            if k == 0:
                inactive = im.copy()
            else:
                inactive = np.delete(im, np.append(active, ignores))
            C = Cvec[inactive]
            Cmax = np.max(np.abs(C))
            if not np.any(drops):
                new = (np.abs(C) >= Cmax)
                C = C[~new]
                new = inactive[new]
                for inew in new:
                    if len(active) > 0:
                        xold = x[:, active]
                    else:
                        xold = None
                    R = _updateRR(x[:, inew], R, xold, lambda_val)
                    if R["rank"] == len(active):
                        nR = np.arange(len(active))
                        R["data"] = R["data"][nR, :][:, nR]
                        R["rank"] = len(active)
                        ignores = np.append(ignores, inew).astype(int)
                        action = np.append(action, -inew)
                    else:
                        if first_in[inew] == 0:
                            first_in[inew] = k
                        active = np.append(active, inew).astype(int)
                        Sign = np.append(Sign, np.sign(Cvec[inew]))
                        action = np.append(action, inew).astype(int)
            else:
                action = -dropid

            Gi1 = scipy.linalg.solve_triangular(R["data"],
                    scipy.linalg.solve_triangular(R["data"], Sign, trans=1))
            A = 1. / np.sqrt((Gi1 * Sign).sum())
            w = A * Gi1
            u1 = np.squeeze(x[:, active].dot(w * d2))
            u2 = np.zeros(m)
            u2[active] = d1 * d2 * w
            u = np.append(u1, u2)
            if lambda_val > 0:
                max_vars = m - len(ignores)
            elif lambda_val == 0:
                max_vars = np.min(m - len(ignores), n - 1)
            if len(active) == max_vars - len(ignores):
                gamhat = Cmax / A
            else:
                mask = np.array([True] * x.shape[1])
                mask[np.append(active, ignores)] = False
                a = (u1.dot(x[:, mask] + d1 * u2[mask])) * d2
                gam = np.append((Cmax - C)/(A - a), (Cmax + C) / (A + a))
                gamhat = np.min([gam[gam > eps].min(), Cmax / A])
                Cdrop = np.append(C - gamhat * a, -C + gamhat * a) - (Cmax - gamhat * A)
            dropid = None
            b1 = beta[active]
            z1 = -b1 / w
            z1_for_min = z1[z1 > eps]
            if len(z1_for_min) > 0:
                zmin = np.min([z1_for_min.min(), gamhat])
            else:
                zmin = gamhat
            if zmin < gamhat:
                gamhat = zmin
                drops = z1 == zmin
            else:
                drops = False

            beta2 = beta.copy()
            beta[active] = beta[active] + gamhat * w
            residuals = residuals - (gamhat * u)
            Cvec = (np.squeeze(residuals[:n].T.dot(x)) + d1 * residuals[n:]) * d2
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
                    R = _downdateR(R, id + 1)
                dropid = active[drops]
                beta[dropid] = 0.
                active = active[~drops]
                Sign = Sign[~drops]
            if (sparse == "varnum") and (len(active) >= paras[1]):
                break
            actions[k] = action
            k += 1
    return beta


def _updateRR(xnew, R, xold, lambda_val, eps=2.22e-16):
    xtx = ((xnew ** 2).sum() + lambda_val) / (1. + lambda_val)
    norm_xnew = np.sqrt(xtx)
    if not R:
        R = {"data": np.reshape(np.array(norm_xnew), newshape=(1, -1)), "rank": 1}
        return R

    Xtx = xnew.dot(xold) / (1. + lambda_val)
    r = scipy.linalg.solve_triangular(R["data"], Xtx, trans=1)
    rpp = norm_xnew ** 2 - (r ** 2).sum()
    rank = R["rank"]
    if rpp <= eps:
        rpp = eps
    else:
        rpp = np.sqrt(rpp)
        rank += 1
    R["data"] = np.vstack([np.vstack([R["data"], np.zeros(R["data"].shape[1])]).T, np.append(r, rpp)]).T
    R["rank"] = rank
    return R


def _downdateR(R, k=None):
    p = R["data"].shape[0]
    if p == 1:
        return None

    if not k:
        k = p
    R["data"] = _delcol(R["data"], np.ones(p), k)[:-1, :]
    R["rank"] = p - 1
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
    lars_fort.delcol(r, int(k), z, int(p), int(dz[0]), int(dz[1]))
    return r
