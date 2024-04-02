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
    April, 2024.
"""

## Library importation.
import os
import glob
import Wave_2D
import numpy as np
import Scripts.Graph as Graph
import Scripts.Errors as Errors
from scipy.io import loadmat
from scipy.io import savemat
from Scripts.TarFile import make_tarfile

def run_example(Holes):
    ## Problem parameters.
    c       = np.sqrt(1/2)                                                          # Wave coefficient.
    cho     = 0                                                                     # Approximation Type (Boundary condition).
    sizes   = [1, 2, 3]                                                             # Size of the clouds to use.
    t       = 2000                                                                  # Number of time-steps.
    Save    = True                                                                  # Should I save the results?

    ## Boundary conditions.
    f = lambda x, y, t, c, cho, r: 0.2*np.exp(-2000*((x - r[0] - c*t)**2 + (y - r[1] - c*t)**2))
                                                                                    # f = 0.2e^(-2000((x - r_x - ct)^2 + (y - r_y - ct)^2))
    g = lambda x, y, t, c, cho, r: 0                                                # g = 0

    # Consolidated path construction
    data_path    = 'Data/{}/'.format('Holes' if Holes else 'Clouds')
    results_path = 'Results/Example 3/{}/'.format('Holes' if Holes else 'Clouds')

    ## Run the example for all the chosen regions.
    for me in sizes:
        cloud       = str(me)

        ## Find and organize all the regions.
        regions_path = glob.glob(f'{data_path}{cloud}/*.mat')
        regions      = sorted([os.path.splitext(os.path.basename(region))[0] for region in regions_path])

        for reg in regions:
            print(f'Region: {reg}, with size: {cloud}')

            # Initial Drop
            region_values = {
                'BAN': [0.1, 0.3],
                'BLU': [0.2, 0.35],
                'CAB': [0.7, 0.15],
                'CUA': [0.8, 0.3],
                'CUI': [0.8, 0.3],
                'DOW': [0.35, 0.15],
                'ENG': [0.3, 0.5],
                'GIB': [0.2, 0.3],
                'HAB': [0.8, 0.8],
                'MIC': [0.3, 0.2],
                'PAT': [0.8, 0.8],
                'TIT': [0.25, 0.3],
                'TOB': [0.7, 0.2],
                'UCH': [0.7, 0.45],
                'VAL': [0.8, 0.3],
                'ZIR': [0.7, 0.25]
            }
            r = np.array(region_values.get(reg, "Unknown region"))

            ## All data is loaded from the file
            mat = loadmat(f'{data_path}{cloud}/{reg}.mat')
            
            ## Node data is saved
            p   = mat['p']
            tt  = mat['tt']
            if tt.min() == 1:
                tt -= 1
            
            ## Wave Equation in 2D computed on a unstructured cloud of points.
            u_ap, u_ex, vec = Wave_2D.Cloud(p, f, g, t, c, cho, r, implicit = True, triangulation = False, tt = tt, lam = 0.5)
            
            if Save:
                folder = os.path.join(results_path, reg)
                os.makedirs(folder, exist_ok = True)

                ## Save the solution on video and graphs.
                Graph.Cloud_1(p, tt, u_ap, save = True, nom = os.path.join(folder, f'{reg}_{cloud}.mp4'))
                Graph.Cloud_Steps_1(p, tt, u_ap, nom = os.path.join(folder, f'{reg}_{cloud}'))

                ## Save the solution un MATLAB format.
                #mdic = {'u_ap': u_ap, 'p': p, 'tt': tt}
                #savemat(os.path.join(folder, f'{reg}_{cloud}.mat'), mdic)
                #make_tarfile(os.path.join(folder, f'{reg}_{cloud}' + '.tar.gz'), os.path.join(folder, f'{reg}_{cloud}.mat'))
            else:
                Graph.Cloud_1(p, tt, u_ap, save = False)

configurations = [
    (False),
    (True)
]

for Holes in configurations:
    print(f'\nComputing numerical solution with Holes = {Holes}.')
    run_example(Holes)
    print("Computation completed.\n")


'''
## Problem parameters.
c       = 1                                                                         # Wave coefficient.
cho     = 0                                                                         # Approximation Type (Boundary condition).
sizes   = [1, 2, 3]                                                                 # Size of the clouds to use.
t       = 4000                                                                      # Number of time-steps.
Holes   = False                                                                     # Should I use clouds with holes?
regular = False                                                                     # Should I use the clouds generated with Dmsh?
Save    = True                                                                      # Should I save the results?
first   = True                                                                      # for the first iteration only.

## Boundary conditions.
f = lambda x, y, t, c, cho, r: 0.2*np.exp(-2000*((x - r[0] - c*t)**2 + (y - r[1] - c*t)**2))
                                                                                    # f = 0.2e^(-2000((x - r_x - ct)^2 + (y - r_y - ct)^2))
g = lambda x, y, t, c, cho, r: 0                                                    # g = 0

## Run the example for all the chosen regions.
for me in sizes:
    cloud = str(me)
    if first:
        # Find all the regions.
        if Holes:
            if regular:
                regions = glob.glob(f'Data/Holes/' + cloud + '/*.mat')
            else:
                regions = glob.glob(f'Data/Holes_rand/' + cloud + '/*.mat')
        else:
            if regular:
                regions = glob.glob(f'Data/Clouds/' + cloud + '/*.mat')
            else:
                regions = glob.glob(f'Data/Clouds_rand/' + cloud + '/*.mat')
            
        regions = sorted([os.path.splitext(os.path.basename(region))[0] for region in regions])
        first = False

    for reg in regions:
        print('Region: ' + reg + ', with size: ' + cloud)

        # Initial Drop
        if reg == 'BAN':
            r = np.array([0.1, 0.3])
        if reg == 'BLU':
            r = np.array([0.2, 0.35])
        if reg == 'CAB':
            r = np.array([0.7, 0.15])
        if reg == 'CUA':
            r = np.array([0.8, 0.3])
        if reg == 'CUI':
            r = np.array([0.8, 0.3])
        if reg == 'DOW':
            r = np.array([0.35, 0.15])
        if reg == 'ENG':
            r = np.array([0.3, 0.5])
        if reg == 'GIB':
            r = np.array([0.2, 0.3])
        if reg == 'HAB':
            r = np.array([0.8, 0.8])
        if reg == 'MIC':
            r = np.array([0.3, 0.2])
        if reg == 'PAT':
            r = np.array([0.8, 0.8])
        if reg == 'TIT':
            r = np.array([0.25, 0.3])
        if reg == 'TOB':
            r = np.array([0.7, 0.2])
        if reg == 'UCH':
            r = np.array([0.7, 0.45])
        if reg == 'VAL':
            r = np.array([0.8, 0.3])
        if reg == 'ZIR':
            r = np.array([0.7, 0.25])
                    
        # All data is loaded from the file
        if Holes:
            if regular:
                mat = loadmat('Data/Holes/' + cloud + '/' + reg + '.mat')
            else:
                mat = loadmat('Data/Holes_rand/' + cloud + '/' + reg + '.mat')
        else:
            if regular:
                mat = loadmat('Data/Clouds/' + cloud + '/' + reg + '.mat')
            else:
                mat = loadmat('Data/Clouds_rand/' + cloud + '/' + reg + '.mat')

        # Node data is saved
        p   = mat['p']
        tt  = mat['tt']
        if tt.min() == 1:
            tt -= 1
        
        ## Wave Equation in 2D computed on a unstructured cloud of points.
        u_ap, u_ex, vec = Wave_2D.Cloud(p, f, g, t, c, cho, r, implicit = True, triangulation = False, tt = tt, lam = 0.5)
        
        if Save:
            if Holes:
                if regular:
                    folder = 'Results/Example 3/Holes/' + reg
                else:
                    folder = 'Results/Example 3/Holes_rand/' + reg
            else:
                if regular:
                    folder = 'Results/Example 3/Clouds/' + reg
                else:
                    folder = 'Results/Example 3/Clouds_rand/' + reg

            if not os.path.exists(folder):
                os.makedirs(folder)
            Graph.Cloud_1(p, tt, u_ap, save = True, nom = folder + '/' + reg + '_' + cloud + '.mp4')
            Graph.Cloud_Steps_1(p, tt, u_ap, nom = folder + '/' + reg + '_' + cloud)
            #file_n = folder + '/' + reg + '_' + cloud + '.mat'
            #mdic = {'u_ap': u_ap, 'p': p, 'tt': tt}
            #savemat(file_n, mdic)
            #make_tarfile(file_n + '.tar.gz', file_n)
        else:
            Graph.Cloud_1(p, tt, u_ap, save = False)

Holes = True                                                                        # Should I use clouds with holes?

## Run the example for all the chosen regions.
for me in sizes:
    cloud = str(me)
    if first:
        # Find all the regions.
        if Holes:
            if regular:
                regions = glob.glob(f'Data/Holes/' + cloud + '/*.mat')
            else:
                regions = glob.glob(f'Data/Holes_rand/' + cloud + '/*.mat')
        else:
            if regular:
                regions = glob.glob(f'Data/Clouds/' + cloud + '/*.mat')
            else:
                regions = glob.glob(f'Data/Clouds_rand/' + cloud + '/*.mat')
            
        regions = sorted([os.path.splitext(os.path.basename(region))[0] for region in regions])
        first = False

    for reg in regions:
        print('Region: ' + reg + ', with size: ' + cloud)

        # Initial Drop
        if reg == 'BAN':
            r = np.array([0.1, 0.3])
        if reg == 'BLU':
            r = np.array([0.2, 0.35])
        if reg == 'CAB':
            r = np.array([0.7, 0.15])
        if reg == 'CUA':
            r = np.array([0.8, 0.3])
        if reg == 'CUI':
            r = np.array([0.8, 0.3])
        if reg == 'DOW':
            r = np.array([0.35, 0.15])
        if reg == 'ENG':
            r = np.array([0.3, 0.5])
        if reg == 'GIB':
            r = np.array([0.2, 0.3])
        if reg == 'HAB':
            r = np.array([0.8, 0.8])
        if reg == 'MIC':
            r = np.array([0.3, 0.2])
        if reg == 'PAT':
            r = np.array([0.8, 0.8])
        if reg == 'TIT':
            r = np.array([0.25, 0.3])
        if reg == 'TOB':
            r = np.array([0.7, 0.2])
        if reg == 'UCH':
            r = np.array([0.7, 0.45])
        if reg == 'VAL':
            r = np.array([0.8, 0.3])
        if reg == 'ZIR':
            r = np.array([0.7, 0.25])
                    
        # All data is loaded from the file
        if Holes:
            if regular:
                mat = loadmat('Data/Holes/' + cloud + '/' + reg + '.mat')
            else:
                mat = loadmat('Data/Holes_rand/' + cloud + '/' + reg + '.mat')
        else:
            if regular:
                mat = loadmat('Data/Clouds/' + cloud + '/' + reg + '.mat')
            else:
                mat = loadmat('Data/Clouds_rand/' + cloud + '/' + reg + '.mat')

        # Node data is saved
        p   = mat['p']
        tt  = mat['tt']
        if tt.min() == 1:
            tt -= 1
        
        ## Wave Equation in 2D computed on a unstructured cloud of points.
        u_ap, u_ex, vec = Wave_2D.Cloud(p, f, g, t, c, cho, r, implicit = True, triangulation = False, tt = tt, lam = 0.5)
        
        if Save:
            mdic = {'u_ap': u_ap, 'p': p, 'tt': tt}
            if Holes:
                if regular:
                    folder = 'Results/Example 3/Holes/' + reg
                else:
                    folder = 'Results/Example 3/Holes_rand/' + reg
            else:
                if regular:
                    folder = 'Results/Example 3/Clouds/' + reg
                else:
                    folder = 'Results/Example 3/Clouds_rand/' + reg

            if not os.path.exists(folder):
                os.makedirs(folder)
            Graph.Cloud_1(p, tt, u_ap, save = True, nom = folder + '/' + reg + '_' + cloud + '.mp4')
            Graph.Cloud_Steps_1(p, tt, u_ap, nom = folder + '/' + reg + '_' + cloud)
            #file_n = folder + '/' + reg + '_' + cloud + '.mat'
            #savemat(file_n, mdic)
            #make_tarfile(file_n + '.tar.gz', file_n)
        else:
            Graph.Cloud_1(p, tt, u_ap, save = False)
'''