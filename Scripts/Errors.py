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

def PolyArea(x,y):
    """
    PolyArea
    Function to calculate the area of a polygon defined by the vertices whose coordinates are stored in $x$ and $y$.
    
    Input:
        x           m x n           Array           Array with the coordinates in x of the vertices of the polygon.
        y           m x n           Array           Array with the coordinates in y of the vertices of the polygon.
    
    Output:
        area                        Float           Area of the polygon.
    """
    ## Area computation.
    area = 0.5*np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))          # Compute the area of the element.
    return area

def Cloud(p, vec, u_ap, u_ex):
    """
    Cloud
    Function to compute the error in a triangulation or an unstructured cloud of points for a problem that depends on time.
    The polygon used to calculate the area is the one defined by all the immediate neighbors of the central node.
    
    Input:
        p           m x 2           Array           Array with the coordinates of the nodes.
        vec         m x nvec        Array           Array with the correspondence of the nvec neighbors of each node.
        u_ap        m x t           Array           Array with the computed solution.
        u_ex        m x t           Array           Array with the theoretical solution.
    
    Output:
        er          t x 1           Array           Mean square error computed on each time step.
    """

    ## Variable initialization.
    m, t = p.shape[0], u_ap.shape[1]                                                # The size of the region.
    er   = np.zeros(t)                                                              # er initialization with zeros.
    area = np.zeros(m)                                                              # area initialization with zeros.

    ## Area computation for each node.
    for i in np.arange(m):
        nvec     = sum(vec[i,:] != -1)                                              # The number of neighbors of the central node.
        polix    = np.zeros([nvec])                                                 # The x-values of the polygon are stored.
        poliy    = np.zeros([nvec])                                                 # The y-values of the polygon are stored.
        nindex   = vec[i, :nvec].astype(int)                                        # Indices of the neighbors.
        polix[:] = p[nindex, 0]                                                     # The x coordinate of the node is stored.
        poliy[:] = p[nindex, 1]                                                     # The y coordinate of the node is stored.
        area[i]  = PolyArea(polix, poliy)                                           # Area computation.

    ## Error computation.
    for k in np.arange(t):                                                          # For each time step.
        err   = np.square(u_ap[:, k] - u_ex[:, k])*area                             # Mean square error computation.
        er[k] = np.sqrt(np.mean(err))                                               # The square root is computed.
    
    return er