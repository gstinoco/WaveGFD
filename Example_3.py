"""
All the codes presented below were developed by:
    Dr. Gerardo Tinoco Guerrero
    Universidad Michoacana de San Nicolás de Hidalgo
    gerardo.tinoco@umich.mx

With the funding of:
    National Council of Humanities, Sciences and Technologies, CONAHCyT (Consejo Nacional de Humanidades, Ciencias y Tecnologías, CONAHCyT). México.
    Coordination of Scientific Research, CIC-UMSNH (Coordinación de la Investigación Científica de la Universidad Michoacana de San Nicolás de Hidalgo, CIC-UMSNH). México
    Aula CIMNE-Morelia. México

Date:
    November, 2022.

Last Modification:
    March, 2024.
"""

## Library importation.
import os
import Wave_2D
import numpy as np
import Scripts.Graph as Graph
import Scripts.Errors as Errors
from scipy.io import loadmat
from scipy.io import savemat
from Scripts.TarFile import make_tarfile

## Problem parameters.
c       = 1                                                                         # Wave coefficient.
cho     = 0                                                                         # Approximation Type (Boundary condition).
regions = ['CAB', 'CUA', 'CUI', 'DOW', 'ENG', 'GIB', 'HAB', 'MIC', 'PAT', 'ZIR']    # Regions to use.
sizes   = [2, 3]                                                                    # Size of the clouds to use.
t       = 2000                                                                      # Number of time-steps.
Holes   = True                                                                      # Should I use clouds with holes?
Save    = True                                                                      # Should I save the results?

## Boundary conditions.
f = lambda x, y, t, c, cho, r: 0.2*np.exp(-2000*((x - r[0] - c*t)**2 + (y - r[1] - c*t)**2))
                                                                                    # f = 0.2e^(-2000((x - r_x - ct)^2 + (y - r_y - ct)^2))
g = lambda x, y, t, c, cho, r: 0                                                    # g = 0

## Run the example for all the chosen regions.
for reg in regions:
    # Initial Drop
    if reg == 'CAB':
        r = np.array([0.5, 0.6])
    if reg == 'CUA':
        r = np.array([0.7, 0.5])
    if reg == 'CUI':
        r = np.array([0.4, 0.6])
    if reg == 'DOW':
        r = np.array([0.4, 0.6])
    if reg == 'ENG':
        r = np.array([0.7, 0.3])
    if reg == 'GIB':
        r = np.array([0.2, 0.4])
    if reg == 'HAB':
        r = np.array([0.8, 0.8])
    if reg == 'MIC':
        r = np.array([0.3, 0.3])
    if reg == 'PAT':
        r = np.array([0.8, 0.8])
    if reg == 'ZIR':
        r = np.array([0.7, 0.5])

    for me in sizes:
        cloud = str(me)
        print('Region: ' + reg + ', with size: ' + cloud)
        
        # All data is loaded from the file
        if Holes:
            mat = loadmat('Data/Holes/' + reg + '_' + cloud + '.mat')
        else:
            mat = loadmat('Data/Clouds/' + reg + '_' + cloud + '.mat')

        # Node data is saved
        p   = mat['p']
        tt  = mat['tt']
        if tt.min() == 1:
            tt -= 1
        
        ## Wave Equation in 2D computed on a unstructured cloud of points.
        u_ap, u_ex, vec = Wave_2D.Cloud(p, f, g, t, c, cho, r, implicit = True, triangulation = True, tt = tt, lam = 0.75)
        
        if Save:
            mdic = {'u_ap': u_ap, 'p': p, 'tt': tt}
            if Holes:
                folder = 'Results/Example 3/Holes/' + reg
                if not os.path.exists(folder):
                    os.makedirs(folder)
                Graph.Cloud_1(p, tt, u_ap, save = True, nom = folder + '/' + reg + '_' + cloud + '.mp4')
                Graph.Cloud_Steps_1(p, tt, u_ap, nom = folder + '/' + reg + '_' + cloud)
                #file_n = folder + '/' + reg + '_' + cloud + '.mat'
                #savemat(file_n, mdic)
                #make_tarfile(file_n + '.tar.gz', file_n)
            else:
                folder = 'Results/Example 3/Clouds/' + reg
                if not os.path.exists(folder):
                    os.makedirs(folder)
                Graph.Cloud_1(p, tt, u_ap, save = True, nom = folder + '/' + reg + '_' + cloud + '.mp4')
                Graph.Cloud_Steps_1(p, tt, u_ap, nom = folder + '/' + reg + '_' + cloud)
                #file_n = folder + '/' + reg + '_' + cloud + '.mat'
                #savemat(file_n, mdic)
                #make_tarfile(file_n + '.tar.gz', file_n)
        else:
            Graph.Cloud_1(p, tt, u_ap, save = False)