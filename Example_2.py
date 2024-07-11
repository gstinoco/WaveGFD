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
    July, 2024.
"""

## Library importation.
import os
import glob
import Wave_2D
import numpy as np
import pandas as pd
import Scripts.Graph as Graph
import Scripts.Errors as Errors

def run_example(Holes):
    ## Problem parameters.
    c       = 1                                                                     # Wave coefficient.
    cho     = 1                                                                     # Approximation Type (Boundary condition).
    sizes   = [1, 2, 3]                                                             # Size of the clouds to use.
    r       = np.array([0, 0])                                                      # No water drop-function.
    t       = 2000                                                                  # Number of time-steps.
    Save    = True                                                                  # Should I save the results?

    ## Boundary conditions.
    f = lambda x, y, t, c, cho, r: np.cos(np.pi*c*t*np.sqrt(2))*np.sin(np.pi*x)*np.sin(np.pi*y)
                                                                                    # f = \cos(\pi c t\sqrt{2})\sin(\pi x)\sin(\pi y)
    g = lambda x, y, t, c, cho, r: -(np.pi*c*t*np.sqrt(2))*np.sin(np.pi*c*t*np.sqrt(2))*np.sin(np.pi*x)*np.sin(np.pi*y)
                                                                                    # g = -(\pi c \sqrt{2})\sin(\pi c t\sqrt{2})\sin(\pi x)\sin(\pi y)

    # Consolidated path construction
    data_path    = 'Data/{}/'.format('Holes' if Holes else 'Clouds')                # Path to look for the data.
    results_path = 'Results/Example 2/{}/'.format('Holes' if Holes else 'Clouds')   # Path to store the results.

    ## Run the example for all the chosen regions.
    for me in sizes:
        cloud       = str(me)

        ## Find and organize all the regions.
        regions_path = glob.glob(f'{data_path}{cloud}/*_p.csv')
        regions      = sorted([os.path.splitext(os.path.basename(region))[0].replace('_p', '') for region in regions_path])

        for reg in regions:
            print(f'Region: {reg}, with size: {cloud}')
            
            ## Load data from CSV files
            p   = pd.read_csv(f'{data_path}{cloud}/{reg}_p.csv', header = None).to_numpy()
            tt  = pd.read_csv(f'{data_path}{cloud}/{reg}_tt.csv', header = None).to_numpy()
            
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

                ## Save the solution into CSV format.
                #df_u_ap = pd.DataFrame(u_ap)
                #df_u_ap.to_csv(os.path.join(folder, f'{reg}_{cloud}_u_ap.csv'), index = False, header = False)
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