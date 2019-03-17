import numpy as np
cimport numpy as cnp


def reflect_target(inp, target):
    cdef cnp.ndarray[cnp.float_t, ndim = 3] inp = inp
    cdef cnp.ndarray[cnp.int64_t, ndim = 2] target = target
    cdef cnp.ndarray[cnp.int64_t, ndim = 2] reflected_target = target

    reflected_target = np.abs(target)
    cdef int dim
    #cdef cnp.ndarray [cnp.bool, ndim = 1] t2 = np.zeros()
    for dim in range(3):
        s = inp.shape[dim]
        r = reflected_target[:, dim]
        while r.max() >= s:  # calculate needed iterations instead of looping
            t2 = r >= s
            t = r[t2]
            t = 2 * s - t - 2
            r[t2] = t
            r = np.abs(r)
        reflected_target[:, dim] = r

    return reflected_target
