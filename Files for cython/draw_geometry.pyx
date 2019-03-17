import tifffile as tifff
import numpy as np

def cube(geom, rad_sphere, ln):
    geom[(ln / 2 - rad_sphere): (ln / 2 + rad_sphere), (ln / 2 - rad_sphere): (ln / 2 + rad_sphere), (
        ln / 2 - rad_sphere): (ln / 2 + rad_sphere)] = 0  # this sets the initialized geom to 16cube walking space
    return geom

def sphere(geom, rad_sphere, ln):
    cdef int s_x, s_y, s_z
    for s_x in xrange(ln):
        for s_y in xrange(ln):
            for s_z in xrange(ln):
                if (s_x - ln / 2) ** 2 + (s_y - ln / 2) ** 2 + (s_z - ln / 2) ** 2 <= ((rad_sphere + 1 / 2) ** 2):
                    geom[[s_x], [s_y], [s_z]] = 0
    return geom


# binarise the geometry to 0 and 1
def binarize_geom(geom_bin, geom, fc):  # binarizes the geometry
    f2 = abs(geom)
    idx = (f2 > fc)
    geom_bin[idx] = 1
    return geom_bin

def importGeom(rad_sphere, ln, filename):
    geom = tifff.imread('cube.tif')
    rad_sphere = geom.shape[1]/ 2  # find the radius from the dimension of the geometry
    ln = rad_sphere * 2 + 6  # set this size for simulated geometries

    # binarise the volume
    geom_bin = geom
    fc = 0
    geom = np.int64(binarize_geom(geom_bin, geom, fc))

    return geom
