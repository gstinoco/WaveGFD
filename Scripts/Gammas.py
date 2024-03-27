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

def Cloud(p, vec, L):
    """
    2D Clouds of Points Gammas Computation.
     
    This function computes the Gamma values for clouds of points, and assemble the K matrix for the computations.
     
    Input:
        p           m x 3           Array           Array with the coordinates of the nodes and a flag for the boundary.
        vec         m x nvec        Array           Array with the correspondence of the 'nvec' neighbors of each node.
        L           5 x 1           Array           Array with the values of the differential operator.
     
     Output:
        K           m x m           Array           K Matrix with the computed Gammas.
    """
    
    ## Variable initialization.
    nvec  = len(vec[0,:])                                                           # The maximum number of neighbors.
    m     = len(p[:,0])                                                             # The total number of nodes.
    K     = np.zeros([m,m])                                                         # K initialization with zeros.
    
    ## Gammas computation and Matrix assembly.
    for i in np.arange(m):                                                          # For each of the nodes.
        if p[i,2] == 0:                                                             # If the node is an inner node.
            nvec = sum(vec[i,:] != -1)                                              # The total number of neighbors of the node.
            dx   = np.zeros([nvec])                                                 # dx initialization with zeros.
            dy   = np.zeros([nvec])                                                 # dy initialization with zeros.
            for j in np.arange(nvec):                                               # For each of the neighbor nodes.
                vec1  = int(vec[i, j])                                              # The neighbor index is found.
                dx[j] = p[vec1, 0] - p[i,0]                                         # dx is computed.
                dy[j] = p[vec1, 1] - p[i,1]                                         # dy is computed.
            M     = np.vstack([[dx], [dy], [dx**2], [dx*dy], [dy**2]])              # M matrix is assembled.
            M     = np.linalg.pinv(M)                                               # The pseudoinverse of matrix M.
            YY    = M@L                                                             # M*L computation.
            Gamma = np.vstack([-sum(YY), YY]).transpose()                           # Gamma values are found.
            K[i,i] = Gamma[0,0]                                                     # The corresponding Gamma for the central node.
            for j in np.arange(nvec):                                               # For each of the neighbor nodes.
                K[i, vec[i,j]] = Gamma[0,j+1]                                       # The corresponding Gamma for the neighbor node.
            
        if p[i,2] == 1:                                                             # If the node is in the boundary.
            K[i,i] = 0                                                              # Central node weight is equal to 0.
            for j in np.arange(nvec):                                               # For each of the neighbor nodes.
                K[i, vec[i,j]] = 0                                                  # Neighbor node weight is equal to 0.
    return K