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
    cho     = 1                                                                     # Approximation Type (Boundary condition).
    sizes   = [1, 2, 3]                                                             # Size of the clouds to use.
    r       = np.array([0, 0])                                                      # No water drop-function.
    t       = 2000                                                                  # Number of time-steps.
    Save    = True                                                                  # Should I save the results?

    ## Boundary conditions.
    f = lambda x, y, t, c, cho, r: np.cos(np.pi*t)*np.sin(np.pi*(x+y))              # f = \cos{\pi t}\sin{\pi(x + y)}
    g = lambda x, y, t, c, cho, r: -np.sin(np.pi*t)*np.sin(np.pi*(x+y))             # g = -\pi\sin{\pi t}\sin{\pi(x + y)}

    # Consolidated path construction
    data_path    = 'Data/{}/'.format('Holes' if Holes else 'Clouds')                # Path to look for the data.
    results_path = 'Results/Example 1/{}/'.format('Holes' if Holes else 'Clouds')   # Path to store the results.

    ## Run the example for all the chosen regions.
    for me in sizes:
        cloud       = str(me)

        ## Find and organize all the regions.
        regions_path = glob.glob(f'{data_path}{cloud}/*.mat')
        regions      = sorted([os.path.splitext(os.path.basename(region))[0] for region in regions_path])

        for reg in regions:
            print(f'Region: {reg}, with size: {cloud}')
            
            ## All data is loaded from the file
            mat = loadmat(f'{data_path}{cloud}/{reg}.mat')
            
            ## Node data is saved
            p   = mat['p']
            tt  = mat['tt']
            if tt.min() == 1:
                tt -= 1
            
            ## Wave Equation in 2D computed on a unstructured cloud of points.
            u_ap, u_ex, vec = Wave_2D.Cloud(p, f, g, t, c, cho, r, implicit = True, triangulation = False, tt = tt, lam = 0.5)
            
            ## Error computation.
            er1 = Errors.Cloud(p, vec, u_ap, u_ex)
            print(f'\tThe mean square error is:\t\t{er1.mean()}')
            
            if Save:
                folder = os.path.join(results_path, reg)
                os.makedirs(folder, exist_ok = True)

                ## Save the solution on video and graphs.
                Graph.Cloud(p, tt, u_ap, u_ex, save = True, nom = os.path.join(folder, f'{reg}_{cloud}.mp4'))
                Graph.Cloud_Steps(p, tt, u_ap, u_ex,nom = os.path.join(folder, f'{reg}_{cloud}'))

                ## Save the solution un MATLAB format.
                #mdic = {'u_ap': u_ap, 'p': p, 'tt': tt}
                #savemat(os.path.join(folder, f'{reg}_{cloud}.mat'), mdic)
                #make_tarfile(os.path.join(folder, f'{reg}_{cloud}' + '.tar.gz'), os.path.join(folder, f'{reg}_{cloud}.mat'))
            else:
                Graph.Cloud(p, tt, u_ap, u_ex, save = False)

## Holes configurations to run several examples.
configurations = [
    (False),
    (True)
]

## Run the examples.
for Holes in configurations:
    print(f'\nComputing numerical solution with Holes = {Holes}.')
    run_example(Holes)
    print("Computation completed.\n") 