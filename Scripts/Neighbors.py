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

def Triangulation(p, tt, nvec):
    """
    Triangulation
    Function to find the neighbor nodes in a triangulation.
    
    Input:
        p           m x 2           double          Array with the coordinates of the nodes.
        tt          n x 3           double          Array with the correspondence of the n triangles.
        nvec                        integer         Maximum number of neighbors.
    
    Output:
        vec         m x nvec        double          Array with matching neighbors of each node.
    """

    ## Variable initialization.
    m   = len(p[:,0])                                                               # The size if the triangulation is obtained.
    vec = np.zeros([m, nvec], dtype=int)-1                                          # The array for the neighbors is initialized.

    ## Neighbor search.
    for i in np.arange(m):                                                          # For each of the nodes.
        kn    = np.argwhere(tt == i)                                                # Search in which triangles the node appears.
        vec2  = np.setdiff1d(tt[kn[:,0]], i)                                        # Neighbors are stored inside vec2.
        vec2  = np.vstack([vec2])                                                   # Convert vec2 to a column.
        nvec2 = sum(vec2[0,:] != -1)                                                # The number of neighbors of the node is calculated.
        nnvec = np.minimum(nvec, nvec2)                                             # The real number of neighbors.
        for j in np.arange(nnvec):                                                  # For each of the nodes.
            vec[i,j] = vec2[0,j]                                                    # Neighbors are saved.
    return vec

def Cloud(p, nvec):
    """
    Clouds
    Routine to find the neighbor nodes in a cloud of points generated with dmsh on Python.
    
    Input:
        p           m x 3           Array           Array with the coordinates of the nodes and a flag for the boundary.
        nvec                        integer         Maximum number of neighbors.
    
    Output:
        vec         m x nvec        double          Array with matching neighbors of each node.
    """

    ## Variable initialization.
    m    = len(p[:,0])                                                              # The size if the triangulation is obtained.
    vec  = np.zeros([m, nvec], dtype=int) - 1                                       # The array for the neighbors is initialized.
    dmin = np.zeros([m, 1]) + 1                                                     # dmin initialization with a "big" value.

    ## Delta computation.
    for i in np.arange(m):                                                          # For each of the nodes.
        x    = p[i,0]                                                               # x coordinate of the central node.
        y    = p[i,1]                                                               # y coordinate of the central node.
        for j in np.arange(m):                                                      # For all the nodes.
            if i != j:                                                              # If the the node is different to the central one.
                x1 = p[j,0]                                                         # x coordinate of the possible neighbor.
                y1 = p[j,1]                                                         # y coordinate of the possible neighbor.
                d  = np.sqrt((x - x1)**2 + (y - y1)**2)                             # Distance from the possible neighbor to the central node.
                dmin[i] = min(dmin[i],d)                                            # Look for the minimum distance.
    dist = (3/2)*max(max(dmin))

    ## Neighbor search.
    for i in np.arange(m):                                                          # For each of the nodes.
        x    = p[i,0]                                                               # x coordinate of the central node.
        y    = p[i,1]                                                               # y coordinate of the central node.
        temp = 0                                                                    # Temporal variable as a counter.
        for j in np.arange(m):                                                      # For all the interior nodes.
            if i != j:                                                              # Check that we are not working with the central node.
                x1 = p[j,0]                                                         # x coordinate of the possible neighbor.
                y1 = p[j,1]                                                         # y coordinate of the possible neighbor.
                d  = np.sqrt((x - x1)**2 + (y - y1)**2)                             # Distance from the possible neighbor to the central node.
                if d < dist:                                                        # If the distance is smaller or equal to the tolerance distance.
                    if temp < nvec:                                                 # If the number of neighbors is smaller than nvec.
                        vec[i,temp] = j                                             # Save the neighbor.
                        temp       += 1                                             # Increase the counter by 1.
                    else:                                                           # If the number of neighbors is greater than nvec.
                        x2 = p[vec[i,:],0]                                          # x coordinates of the current neighbor nodes.
                        y2 = p[vec[i,:],1]                                          # y coordinates of the current neighbor nodes.
                        d2 = np.sqrt((x - x2)**2 + (y - y2)**2)                     # The total distance from all the neighbors to the central node.
                        I  = np.argmax(d2)                                          # Look for the greatest distance.
                        if d < d2[I]:                                               # If the new node is closer than the farthest neighbor.
                            vec[i,I] = j                                            # The new neighbor replace the farthest one.
    return vec 