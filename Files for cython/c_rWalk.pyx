import numpy as np
cimport numpy as cnp
from random import choice

def drinkandwalk(n, x_pos, y_pos, z_pos, pS_an, pV_an, geometry, WalkSteps, mcubesurf):
    # set initial position of each walker from uniform distribution about centre of the geometry
    # RandomExcitation:  # just to reduce an "if" statement from the proc    

    # random walk
    step = [(0, 1, 0, "right"), (0, -1, 0, "left"), (1, 0, 0, "down"), (-1, 0, 0, "up"), (0, 0, -1, "in"),
            (0, 0, 1, "out")]

    cdef int i
    cdef double lsa
    cdef int Wsteps = WalkSteps
    cdef double pS_a = pS_an
    cdef double pV_a = pV_an
    cdef cnp.ndarray[cnp.int64_t, ndim=3] geom = geometry
    cdef cnp.ndarray[cnp.float64_t, ndim=3] g = mcubesurf

    '''
    lsa_calc = {
            'right': (1/4)* g[x_pos[i], y_pos[i+1], z_pos[i+1]] + (1/4)* g[x_pos[i], y_pos[i], z_pos[i]] + (1/4)* g[x_pos[i], y_pos[i], z_pos[i+1]]+(1/4)* g[x_pos[i], y_pos[i+1], z_pos[i]],
            'left': (1/4)* g[x_pos[i+1], y_pos[i+1], z_pos[i+1]] + (1/4)* g[x_pos[i+1], y_pos[i], z_pos[i+1]] + (1/4)* g[x_pos[i+1], y_pos[i+1], z_pos[i]]+(1/4)* g[x_pos[i+1], y_pos[i], z_pos[i]],
            'down': (1/4)* g[x_pos[i+1], y_pos[i], z_pos[i+1]] + (1/4)* g[x_pos[i+1], y_pos[i], z_pos[i]] + (1/4)* g[x_pos[i], y_pos[i], z_pos[i]]+(1/4)* g[x_pos[i], y_pos[i+1], z_pos[i+1]],
            'up': (1/4)* g[x_pos[i+1], y_pos[i+1], z_pos[i+1]] + (1/4)* g[x_pos[i], y_pos[i+1], z_pos[i]] + (1/4)* g[x_pos[i], y_pos[i+1], z_pos[i+1]]+(1/4)* g[x_pos[i+1], y_pos[i+1], z_pos[i]],
            'in': (1/4)* g[x_pos[i+1], y_pos[i+1], z_pos[i]] + (1/4)* g[x_pos[i], y_pos[i], z_pos[i]] + (1/4)* g[x_pos[i+1], y_pos[i], z_pos[i+1]]+(1/4)* g[x_pos[i], y_pos[i+1], z_pos[i]],
            'out': (1/4)* g[x_pos[i+1], y_pos[i+1], z_pos[i+1]] + (1/4)* g[x_pos[i], y_pos[i], z_pos[i+1]] + (1/4)* g[x_pos[i+1], y_pos[i], z_pos[i+1]]+(1/4)* g[x_pos[i], y_pos[i+1], z_pos[i+1]],
            }
    '''
    for i in range(Wsteps):
        move = choice(step)
        x_pos += move[0]
        y_pos += move[1]
        z_pos += move[2]
        print 'x_pos: ', x_pos
        # Reflects back if there is a wall and the annihilation probability is less
        # when in contact surface relaxation
        #print 'geometry', (geom[x_pos[i+1], y_pos[i+1], z_pos[i+1]])
        # if geom[x_pos[i+1], y_pos[i+1], z_pos[i+1]] != 0:

        #lsa = lsa_calc[move[3]]

            #if (lsa*pS_a) > np.random.random():


        #  if (geom[x_pos[i+1], y_pos[i+1], z_pos[i+1]] != 0) and ((g[x_pos[i], y_pos[i], z_pos[i]])*pS_a) > np.random.random():
            #print 'at position', i
           # print 'mc_surf', (f[x_pos[i], y_pos[i], z_pos[i]])
        #  test annihilation at the surface
        if (g[x_pos, y_pos, z_pos] != 0) and (pS_a > np.random.random()):
            print 'step', i
            print 'direction', move[3]
            print 'coordinates', x_pos, y_pos, z_pos
            print 'geometryVal', geom[x_pos, y_pos, z_pos]
            print 'MC_Val', g[x_pos, y_pos, z_pos]
            #print 'localSurfaceCorr', lsa
            return i

        elif geom[x_pos, y_pos, z_pos] != 0:
            print 'refllect'
            x_pos -= move[0]
            y_pos -= move[1]
            z_pos -= move[2]

        # Volume relaxation: the particle dies in the volume when walking
        if pV_a > np.random.random():
            return i

    # print 'Alive', n, i
    return i
