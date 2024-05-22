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
import numpy as np
import Scripts.Gammas as Gammas
import Scripts.Neighbors as Neighbors

def Cloud(p, f, g, t, c, cho, r, triangulation = False, tt = None, implicit = False, lam = 0.5):
    '''
    Numerical solution of the 2D wave equation on irregular domains using a Meshless Generalized Finite Difference Scheme.
    
    This function calculates an approximate solution to the 2D wave equation on irregular domains using a Meshless Generalized
    Finite Difference Scheme. It handles boundary conditions, initial conditions, and neighbor search for the nodes.

    The problem to solve is:
    
    \frac{\partial^2 u}{\partial t^2} = c^2\nabla^2 u$
    
    Input:
        p           m x 2           ndarray         Array with the coordinates of the nodes.
        f                           function        Function declared with the boundary condition.
        g                           function        Function declared with the boundary condition.
        t                           int             Number of time steps to be considered.
        c                           float           Wave propagation velocity.
        cho                         int             Approximation Type.
                                                        (0 for zero boundary condition).
                                                        (1 for function boundary condition).
        r           1 x 2           ndarray         Coordinates of the water drop-like function.
        triangulation               bool            Select whether or not there is a triangulation available.
                                                        True: Triangulation available.
                                                        False: No triangulation available (Default).
        tt          m x 3           ndarray         Array with the triangulation indexes.
        implicit                    bool            Select whether or not use an implicit scheme.
                                                        True: Implicit scheme used.
                                                        False: Explicit scheme used (Default).
        lam                         float           Lambda parameter for the implicit scheme.
                                                        Must be between 0 and 1 (Default: 0.5).
    
    Output:
        u_ap        m x t           ndarray         Array with the approximation computed by the routine.
        u_ex        m x t           ndarray         Array with the theoretical solution.
        vec         m x o           ndarray         Array with the correspondence of the o neighbors of each node.  
    '''
    
    ## Variable initialization.
    m      = len(p[:, 0])                                                           # The total number of nodes is calculated.
    nvec   = 8                                                                      # Maximum number of neighbors for each node.
    T      = np.linspace(0, 1, t)                                                   # Time discretization.
    dt     = T[1] - T[0]                                                            # dt computation.
    u_ap   = np.zeros([m, t])                                                       # u_ap initialization with zeros.
    u_ex   = np.zeros([m, t])                                                       # u_ex initialization with zeros.
    cdt    = (c**2)*(dt**2)                                                         # cdt is equals to c^2 dt^2.
    boun_n = (p[:, 2] == 1) | (p[:, 2] == 2)                                        # Save the boundary nodes.
    inne_n = p[:, 2] == 0                                                           # Save the inner nodes.

    ## Boundary conditions.
    if cho == 1:                                                                    # Approximation Type selection.
        for k in np.arange(t):                                                      # For each time step.
            u_ap[boun_n, k] = f(p[boun_n, 0], p[boun_n, 1], T[k], c, cho, r)        # The boundary condition is assigned.

    ## Initial condition.
    u_ap[:, 0] = f(p[:, 0], p[:, 1], T[0], c, cho, r)                               # The initial condition is assigned.
    
    ## Neighbor search.
    if triangulation == True:                                                       # If there are triangles available.
        vec = Neighbors.Triangulation(p, tt, nvec)                                  # Neighbor search with the proper routine.
    else:                                                                           # If there are no triangles available.
        vec = Neighbors.Cloud(p, nvec)                                              # Neighbor search with the proper routine.

    ## Gamma computation.
    L = np.vstack([[0], [0], [2*cdt], [0], [2*cdt]])                                # The values of the differential operator are assigned.
    K = Gammas.Cloud(p, vec, L)                                                     # K computation with the required Gammas.

    if implicit == False:                                                           # For the explicit scheme.
        K1 = np.identity(m)                                                         # Implicit formulation of K for k = 1.
        K2 = np.identity(m) + (1/2)*K                                               # Implicit formulation of K for k = 1.
        K3 = np.identity(m)                                                         # Implicit formulation of K for k = 2, ..., t.
        K4 = 2*np.identity(m) + K                                                   # Implicit formulation of K for k = 2, ..., t.
    else:                                                                           # For the implicit scheme.
        K1 = np.linalg.pinv(np.identity(m) - (1 - lam)*(1/2)*K)                     # Implicit formulation of K for k = 1.
        K2 = np.identity(m) + lam*(1/2)*K                                           # Implicit formulation of K for k = 1.
        K3 = np.linalg.pinv(np.identity(m) - (1 - lam)*K)                           # Implicit formulation of K for k = 2, ..., t.
        K4 = 2*np.identity(m) + lam*K                                               # Implicit formulation of K for k = 2, ..., t.

    ## Generalized Finite Differences Method
    for k in np.arange(1, t):                                                       # For al time levels.
        if k == 1:                                                                  # For the first time level.
            un = K1@(K2@u_ap[:, k - 1] + dt*g(p[:, 0], p[:, 1], T[k], c, cho, r))   # The new time-level is computed.
            u_ap[inne_n, k] = un[inne_n]                                            # Save the computed solution.
        else:                                                                       # For all the other time levels.
            un = K3@(K4@u_ap[:, k - 1] - u_ap[:, k - 2])                            # The new time-level is computed.
            u_ap[inne_n, k] = un[inne_n]                                            # Save the computed solution.                

    ## Theoretical Solution
    for k in np.arange(t):                                                          # For all the time steps.
        u_ex[:, k] = f(p[:, 0], p[:, 1], T[k], c, cho, r)                           # The theoretical solution is computed.

    return u_ap, u_ex, vec