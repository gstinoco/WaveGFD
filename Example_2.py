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
cho     = 1                                                                         # Approximation Type (Boundary condition).
regions = ['CAB', 'CUA', 'CUI', 'DOW', 'ENG', 'GIB', 'HAB', 'MIC', 'PAT', 'ZIR']    # Regions to use.
sizes   = [2, 3]                                                                    # Size of the clouds to use.
r       = np.array([0, 0])                                                          # No water drop-function.
t       = 2000                                                                      # Number of time-steps.
Holes   = True                                                                      # Should I use clouds with holes?
Save    = True                                                                      # Should I save the results?

## Boundary conditions.
f = lambda x, y, t, c, cho, r: np.cos(np.pi*c*t*np.sqrt(2))*np.sin(np.pi*x)*np.sin(np.pi*y)
                                                                                    # f = \cos(\pi c t\sqrt{2})\sin(\pi x)\sin(\pi y)
g = lambda x, y, t, c, cho, r: -(np.pi*c*t*np.sqrt(2))*np.sin(np.pi*c*t*np.sqrt(2))*np.sin(np.pi*x)*np.sin(np.pi*y)
                                                                                    # g = -(\pi c \sqrt{2})\sin(\pi c t\sqrt{2})\sin(\pi x)\sin(\pi y)

## Run the example for all the chosen regions.
for reg in regions:    
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
        
        ## Error computation.
        er1 = Errors.Cloud(p, vec, u_ap, u_ex)
        print('\tThe mean square error is:\t\t', er1.mean())
        
        if Save:
            mdic = {'u_ap': u_ap, 'u_ex': u_ex, 'p': p, 'tt': tt}
            if Holes:
                folder = 'Results/Example 2/Holes/' + reg
                if not os.path.exists(folder):
                    os.makedirs(folder)
                Graph.Cloud(p, tt, u_ap, u_ex, save = True, nom = folder + '/' + reg + '_' + cloud + '.mp4')
                Graph.Cloud_Steps(p, tt, u_ap, u_ex, nom = folder + '/' + reg + '_' + cloud)
                #file_n = folder + '/' + reg + '_' + cloud + '.mat'
                #savemat(file_n, mdic)
                #make_tarfile(file_n + '.tar.gz', file_n)
            else:
                folder = 'Results/Example 2/Clouds/' + reg
                if not os.path.exists(folder):
                    os.makedirs(folder)
                Graph.Cloud(p, tt, u_ap, u_ex, save = True, nom = folder + '/' + reg + '_' + cloud + '.mp4')
                Graph.Cloud_Steps(p, tt, u_ap, u_ex, nom = folder + '/' + reg + '_' + cloud)
                #file_n = folder + '/' + reg + '_' + cloud + '.mat'
                #savemat(file_n, mdic)
                #make_tarfile(file_n + '.tar.gz', file_n)
        else:
            Graph.Cloud(p, tt, u_ap, u_ex, save = False)