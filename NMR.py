__author__ = 'Diwaker Jha'
import time

import matplotlib.pyplot as plt
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from c_marchSurf import MC_Surface
import c_rWalk as rWalk
from itertools import permutations
import draw_geometry
import RemoteException
import scipy.io
import sys

# select which geometry to use
importedGeom = 0
simul_GeomCube = 1
simul_GeomSphere = 0

# select what kind of excitation, point will start all particles from one point.
pointExcitation = 0
RandomExcitation = 1
eachcellExcitation = 0


# number of layers of solid border (1) enclosing the walking volume with zeros
border = int(4)

# select if to include marching cube surface area
marchCubeSurf = 1

if pointExcitation:
    excite = 'pointExcitation'
if RandomExcitation:
    excite = 'randomExcitation'
if eachcellExcitation:
    excite = 'eachCellExcitation'

# number of walkers in the system
total_nof_Walkers = np.int(5)


###################################################################
# setting the physical parameters
# half the size of geometry (cubic assumption)
# if rad_sphere_init = 1 then delta_r = 1.2620312500000002e-08 = ndr
D = 3
R_0 = float(8.077e-5)  # Maximum radius.
# Step size in radius. Number of steps in time is inversely proportional to the square of this.
# Delta_r = np.float64(4.0385e-6)
# Delta_r = float(rad_sphere_init * ndr)
Delta_r = float(4.0385e-06)  # (4.092355980797505e-06)  # 4.0385e-06

D_0 = float(2.1e-9)  # np.float64(np.divide(1, R_0))  # Diffusion constant.
T_2b = float(1e9)  # (1e9) Volume relaxation characteristic time.
rho_0 = float(10)  # Surface relaxation strength
rho = float(2.60e-5)  # comment
# rho = np.divide((rho_0 * D_0), R_0)

g = float(1)  # ratio of calculated to true surface areaNMRv2.0.py
# Delta t
Delta_t = float((Delta_r ** 2) / (2 * 3 * D_0))
# Dimensionless time.
t_max = float(3.236)

# Probabilities
# My derived result for D=3 per boundary, i.e., without the S factor.
# Probability (denoted \gamma and \delta in the literature!!!) of annihilation when encountering the surface.
pS_an = g * Delta_r * rho / D_0  # g is the global factor

# Probability of volume relaxation
pV_an = Delta_t / T_2b

# number of neighboring faces!
S = float(1)

# number of steps a particle walks
WalkSteps = np.int(np.rint(t_max / Delta_t))
# print 'calculated number of walkers:', np.int(np.rint(np.divide(t_max, Delta_t)))
# WalkSteps = 9000


#############################################################

# initialize the vector to store the position of the steps.  # try to remove this, memory is not needed!
# x_pos = np.zeros(WalkSteps + 1, dtype=np.int)
# y_pos = np.zeros(WalkSteps + 1, dtype=np.int)
# z_pos = np.zeros(WalkSteps + 1, dtype=np.int)

x_pos = int(0), y_pos = int(0), z_pos = int(0)


rndSeed = np.random.randint(1, total_nof_Walkers)

rad_sphere = np.int(
    np.rint(R_0 / Delta_r))  # this number is radius for sphere and twice of it is dimension of cube


# rad_sphere = 20

ln = rad_sphere * 2 + border  # set this size for simulated geometries
geom = np.ones((ln, ln, ln), dtype=np.int)  # this is the initialized geometry

# Import Geometry
if importedGeom:
    title = 'imported Geometry'

    # load geometry
    geom = draw_geometry.importGeom(rad_sphere, ln, 'cube.tif')

# CUBE
if simul_GeomCube:
    title = 'Inside a cube with ' + str(total_nof_Walkers) + ' walkers and ' + excite
    print "Dimension of cube / px: ", rad_sphere * 2
    geom = draw_geometry.cube(geom, rad_sphere, ln)

# SPHERE
if simul_GeomSphere:
    print "solving in a sphere of radius px: ", rad_sphere
    title = 'Inside a sphere with ' + str(total_nof_Walkers) + ' walkers and ' + excite

# retrieve marching cube surface values for each points in 3D geometry
if marchCubeSurf:
    mc_vol = np.zeros(geom.shape)
    mc_vol = MC_Surface(geom, mc_vol, 0.0)
    #mc_vol =  np.roll(mc_vol, 1, axis=0)
    #mc_vol =  np.roll(mc_vol, 1, axis=1)
    #mc_vol =  np.roll(mc_vol, 1, axis=2)


# starting point for the particles.
if eachcellExcitation:
    InitPos = np.asanyarray(list(permutations(range(rad_sphere * 2), 3)), dtype=int)
    InitPos = InitPos[:total_nof_Walkers]

elif RandomExcitation:
    np.random.seed(rndSeed)
    print 'randomSeedVal= ', rndSeed
    InitPos = np.asanyarray(
        np.random.uniform((border / 2), 2 * rad_sphere + (border / 2), (total_nof_Walkers, 3)),
        dtype=int)  # this is one random number for all axis

elif pointExcitation:
    x_pos[0] = ln / 2  # ln is the dimension of the simulated geometry, so the initial action starts at center
    y_pos[0] = ln / 2
    z_pos[0] = ln / 2


@RemoteException.showError
def do_rwalk(n):
    death = rWalk.drinkandwalk(n, x_pos, y_pos, z_pos, pS_an, pV_an, geom, WalkSteps, mc_vol)
    return death

################ Check for the time #############
start_time = time.time()
print 'WalkSteps=', WalkSteps, '   FullGeometrySize=', ln, '    radiusforDOFinGeom=', rad_sphere, '    totalNumberofWalkers=', total_nof_Walkers

################ Pool the job to the processors #############
if __name__ == '__main__':
    cntinitialPos = int(0)
    n = int(0)
    procPool = Pool()
    rms3D = procPool.amap(do_rwalk, range(total_nof_Walkers))
    TimeVectorFinal = np.asanyarray(rms3D.get())

print("time", time.time() - start_time)

################ find where the particle died ##############
num_Walkers = np.zeros(WalkSteps, dtype=np.int) + total_nof_Walkers

for r in range(len(TimeVectorFinal)):
    if TimeVectorFinal[r] != 0:
        num_Walkers[TimeVectorFinal[r]:] -= 1

############## Construct time axis ########################
num_Walkers = num_Walkers[:-1]
t_vector = Delta_t * np.arange(0, WalkSteps - 1)  # this -1 is to kill the last zero walker jump in the curve.

no_of_Walkers_at_t_vector_end = num_Walkers[-1]

norm_no_of_trajectories = num_Walkers / float(total_nof_Walkers)
# norm_no_of_trajectories = norm_no_of_trajectories[:-1]

#######################  Analytical Solution for cube #################################################
#######################################################################################################

# 38 eigenvalues for rho_0 = 10:
'''
lambda_1D_vector = np.float64(1.0e+04) * np.float64(
    [0.000204166950895, 0.001853992580922, 0.005224557087069, 0.010404535687019, 0.017461470328708, 0.026436682746143,
     0.037353425450879, 0.050224612280050, 0.065057557318040, 0.081856539874838, 0.100624155070262, 0.121362032760579,
     0.144071230412063, 0.168752454683257, 0.195406190509948, 0.224032778685422, 0.254632463865127, 0.287205425079669,
     0.321751795612826, 0.358271676245627, 0.396765144263402, 0.437232259697254, 0.479673069723981, 0.524087611817032,
     0.570475916035911, 0.618838006711962, 0.669173903705042, 0.721483623351098, 0.775767179184252, 0.832024582492524,
     0.890255842749454, 0.950460967952233, 1.012639964888733, 1.076792839349967, 1.142919596300355, 1.211020240015063,
     1.281094774191502, 1.353143202040372])
     '''

# 50 eigenvalues for rho_0 = 1:
lambda_1D_vector = 10000 * np.float64(
    [0.000074017388440, 0.001173486182994, 0.004143880784757, 0.009080821420922, 0.015990328897383, 0.024873342660260,
     0.035730110217720, 0.048560718804045, 0.063365205406791, 0.080143587854663, 0.098895875592454, 0.119622073999992,
     0.142322186323129, 0.166996214613980, 0.193644160221264, 0.222266024060875, 0.252861806772410, 0.285431508813447,
     0.319975130518330, 0.356492672135938, 0.394984133854622, 0.435449515819041, 0.477888818141779, 0.522302040911520,
     0.568689184198882, 0.617050248060660, 0.667385232542952, 0.719694137683488, 0.773976963513384, 0.830233710058506,
     0.888464377340423, 0.948668965377416, 1.010847474184783, 1.074999903775677, 1.141126254161282, 1.209226525351270,
     1.279300717353873, 1.351348830176362, 1.425370863824953, 1.501366818305158, 1.579336693621785, 1.659280489778994,
     1.741198206780581, 1.825089844629773, 1.910955403329559, 1.998794882882490, 2.088608283290952, 2.180395604557010,
     2.274156846682459, 2.369892009669071])

q = 1  # uniform initial conditions

c_1D_vector_uni = (np.sin(np.sqrt(lambda_1D_vector)) / np.sqrt(lambda_1D_vector)) ** q * (
    np.sin(np.sqrt(lambda_1D_vector))) / (
                      np.cos(np.sqrt(lambda_1D_vector)) * (np.sin(np.sqrt(lambda_1D_vector))) + np.sqrt(
                          lambda_1D_vector))

q = 0  # uniform initial conditions

c_1D_vector_central = (np.sin(np.sqrt(lambda_1D_vector)) / np.sqrt(lambda_1D_vector)) ** q * (
    np.sin(np.sqrt(lambda_1D_vector))) / (
                          np.cos(np.sqrt(lambda_1D_vector)) * (np.sin(np.sqrt(lambda_1D_vector))) + np.sqrt(
                              lambda_1D_vector))

tau_max = np.divide(D_0, (R_0 ** 2)) * t_max

#### Initialize the vectors to store magnetization and physical time ##############################

tau = np.linspace(0, tau_max, 1e2)
M_uni_1D = np.zeros(len(tau))
M_central_1D = np.zeros(len(tau))
M_uni_1D_physical_time = np.zeros(len(t_vector))
M_central_1D_physical_time = np.zeros(len(t_vector))

for j in range(len(lambda_1D_vector)):
    # Analytic solution with dimensionless time axes.
    M_uni_1D += 2 * c_1D_vector_uni[j] * np.exp(-lambda_1D_vector[j] * tau)
    M_central_1D += 2 * c_1D_vector_central[j] * np.exp(-lambda_1D_vector[j] * tau)

    # Analytic solution with physical time axes.
    M_uni_1D_physical_time += 2 * c_1D_vector_uni[j] * np.exp(-lambda_1D_vector[j] * t_vector * D_0 / R_0 ** 2)
    M_central_1D_physical_time += 2 * c_1D_vector_central[j] * np.exp(-lambda_1D_vector[j] * t_vector * D_0 / R_0 ** 2)

#### 3D magnetization
M_uni_3D = (M_uni_1D ** 3)
M_central_3D = M_central_1D ** 3

M_uni_3D_physical_time = M_uni_1D_physical_time ** 3
M_central_3D_physical_time = M_central_1D_physical_time ** 3

########################################################################################################
########## Plots #######################################################################################

# Numerical Magnetization
fig1 = plt.figure(1)
plt.plot(t_vector, norm_no_of_trajectories, 'bo', t_vector, 1 - 3 * rho / R_0 * t_vector, 'r--')

plt.yticks(np.arange(0, 1 + 0.1, 0.1))
plt.axis((0, t_max, 0, 1))
plt.xlabel('Physical time [s]', fontsize=20)
plt.ylabel('Numerical Magnetization', fontsize=20)
plt.box(on=1)

# Dimensionless time
fig2 = plt.figure(2)
plt.plot(tau, M_uni_3D, 'bo', tau, M_central_3D, 'go')

plt.yticks(np.arange(0, 1 + 0.1, 0.1))
plt.axis((0, tau_max, 0, 1))
plt.xlabel('Dimensionless time', fontsize=20)
plt.ylabel('Analytical Magnetization', fontsize=20)
plt.box(on=1)

# Physical time
fig3 = plt.figure(3)
plt.plot(t_vector, norm_no_of_trajectories, 'b', t_vector, M_uni_3D_physical_time, 'g')

plt.yticks(np.arange(0, 1 + 0.1, 0.1))
plt.axis((0, t_max, 0, 1))
plt.xlabel('Physical time [s]', fontsize=20)
plt.ylabel('Magnetization', fontsize=20)
plt.box(on=1)


# Physical time
fig4 = plt.figure(4)
plt.plot(t_vector, norm_no_of_trajectories - M_uni_3D_physical_time, 'ro')

# plt.yticks(np.arange(-0.05, 1 + 0.1, 0.05))
plt.axis((0, t_max, -0.03, 0.03))
plt.xlabel('Physical time [s]', fontsize=20)
plt.ylabel('Difference (num-ana, 10^6 Walkers)', fontsize=20)

scipy.io.savemat(
    '/Users/i_diwaker/Documents/NMR_relaxation/cNMRv6/' + 'NMR_' + 'geom=' + str(ln) + '_dr=' + str(Delta_r) + '_walkers=' + str(
        total_nof_Walkers) + '.mat',
    mdict={'p_M_Uni_3D': M_uni_3D, 'tau': tau, 'p_t_vector': t_vector,
           'p_M_uni_3D_physical_time': M_uni_3D_physical_time,
           'numericalMagnetization': norm_no_of_trajectories})

print '1st', norm_no_of_trajectories[0] - M_uni_3D_physical_time[0]
print '2nd', norm_no_of_trajectories[1] - M_uni_3D_physical_time[1]
print '3rd', norm_no_of_trajectories[2] - M_uni_3D_physical_time[2]
print '4rth', norm_no_of_trajectories[3] - M_uni_3D_physical_time[3]
print '5th', norm_no_of_trajectories[4] - M_uni_3D_physical_time[4]

plt.show()

# sys.modules[__name__].__dict__.clear()
