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
    data_path    = 'Data/{}/'.format('Holes' if Holes else 'Clouds')                # Path to look for the data.
    results_path = 'Results/Example 3/{}/'.format('Holes' if Holes else 'Clouds')   # Path to store the results.

    ## Run the example for all the chosen regions.
    for me in sizes:
        cloud       = str(me)

        ## Find and organize all the regions.
        regions_path = glob.glob(f'{data_path}{cloud}/*_p.csv')
        regions      = sorted([os.path.splitext(os.path.basename(region))[0].replace('_p', '') for region in regions_path])

        for reg in regions:
            print(f'Region: {reg}, with size: {cloud}')

            ## Initial Drop
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

            ## Load data from CSV files
            p   = pd.read_csv(f'{data_path}{cloud}/{reg}_p.csv', header = None).to_numpy()
            tt  = pd.read_csv(f'{data_path}{cloud}/{reg}_tt.csv', header = None).to_numpy()
            
            ## Wave Equation in 2D computed on a unstructured cloud of points.
            u_ap, u_ex, vec = Wave_2D.Cloud(p, f, g, t, c, cho, r, implicit = True, triangulation = False, tt = tt, lam = 0.5)
            
            if Save:
                folder = os.path.join(results_path, reg)
                os.makedirs(folder, exist_ok = True)

                ## Save the solution on video and graphs.
                Graph.Cloud_1(p, tt, u_ap, save = True, nom = os.path.join(folder, f'{reg}_{cloud}.mp4'))
                Graph.Cloud_Steps_1(p, tt, u_ap, nom = os.path.join(folder, f'{reg}_{cloud}'))

                ## Save the solution into CSV format.
                #df_u_ap = pd.DataFrame(u_ap)
                #df_u_ap.to_csv(os.path.join(folder, f'{reg}_{cloud}_u_ap.csv'), index = False, header = False)
            else:
                Graph.Cloud_1(p, tt, u_ap, save = False)

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