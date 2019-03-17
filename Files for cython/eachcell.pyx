cimport numpy as cnp
import cython
def eachcell(rad_sphere, num_walkers):

    cdef int o =0
    cdef int p =0
    cdef int q =0
    cdef int cntwalkers = 0
    for o in range(rad_sphere*2):
        for p in range(rad_sphere*2):
            for q in range(rad_sphere*2):
                cntwalkers +=1
                if cntwalkers <=num_walkers:
                    print num_walkers, cntwalkers, o, p, q
                    return o, p, q
                else:
                    return
