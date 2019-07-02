import numpy as np
from numba import njit

@njit
def delcol(r, k, z):
    # r : p by *
    # z : n by nz
    r = r.copy()
    z = z.copy()

    p = r.shape[0]
    (n, nz) = z.shape
    p1 = p - 1

    for i in range(k - 1, p - 1): # k coming in with R-style indexing
        a = r[i, i]
        b = r[i + 1, i]
        if b == 0:
            continue
        if np.abs(b) >= np.abs(a):
            tau = -a / b
            s = 1 / np.sqrt(1 + tau * tau)
            c = s * tau
        else:
            tau = -b / a
            c = 1 / np.sqrt(1 + tau * tau)
            s = c * tau

        r[i, i] = c * a - s * b
        r[i + 1, i] = s * a + c * b

        for j in range(i + 1, p1):
            a = r[i, j]
            b = r[i + 1, j]
            r[i, j] = c * a - s * b
            r[i + 1, j] = s * a + c * b

        for j in range(nz):
            a = z[i, j]
            b = z[i + 1, j]
            z[i, j] = c * a - s * b
            z[i + 1, j] = s * a + c * b

    return r, z

