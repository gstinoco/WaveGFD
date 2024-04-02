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
from scipy.spatial import KDTree

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

def Cloud_old(p, nvec):
    """
    Cloud
    Function to find the neighbor nodes in a cloud of points.
    
    Input:
        p           m x 3           ndarray         Array with the coordinates of the nodes and a flag for the boundary.
        nvec                        integer         Maximum number of neighbors.
    
    Output:
        vec         m x nvec        ndarray         Array with matching neighbors of each node.
    """

    ## Variable initialization.
    m    = len(p[:, 0])                                                             # The size if the triangulation is obtained.
    vec  = np.zeros([m, nvec], dtype = int) - 1                                     # The array for the neighbors is initialized.

    ## Delta computation.
    dist = find_distances(p, mode = 1)

    ## Neighbor search.
    for i in np.arange(m):                                                          # For each of the nodes.
        x    = p[i, 0]                                                              # x coordinate of the central node.
        y    = p[i, 1]                                                              # y coordinate of the central node.
        temp = 0                                                                    # Temporal variable as a counter.
        for j in np.arange(m):                                                      # For all the interior nodes.
            if i != j:                                                              # Check that we are not working with the central node.
                x1 = p[j,0]                                                         # x coordinate of the possible neighbor.
                y1 = p[j,1]                                                         # y coordinate of the possible neighbor.
                d  = np.sqrt((x - x1)**2 + (y - y1)**2)                             # Distance from the possible neighbor to the central node.
                if d < dist:                                                        # If the distance is smaller or equal to the tolerance distance.
                    if temp < nvec:                                                 # If the number of neighbors is smaller than nvec.
                        vec[i, temp] = j                                            # Save the neighbor.
                        temp       += 1                                             # Increase the counter by 1.
                    else:                                                           # If the number of neighbors is greater than nvec.
                        x2 = p[vec[i, :], 0]                                        # x coordinates of the current neighbor nodes.
                        y2 = p[vec[i, :], 1]                                        # y coordinates of the current neighbor nodes.
                        d2 = np.sqrt((x - x2)**2 + (y - y2)**2)                     # The total distance from all the neighbors to the central node.
                        I  = np.argmax(d2)                                          # Look for the greatest distance.
                        if d < d2[I]:                                               # If the new node is closer than the farthest neighbor.
                            vec[i,I] = j                                            # The new neighbor replace the farthest one.
    return vec

def Cloud(p, nvec):
    """
    Cloud
    Function to find the neighbor nodes in a cloud of points.
    
    Input:
        p           m x 3           ndarray         Array with the coordinates of the nodes and a flag for the boundary.
        nvec                        integer         Maximum number of neighbors.
    
    Output:
        vec         m x nvec        ndarray         Array with matching neighbors of each node.
    """

    ## Delta computation.
    dist = find_distances(p, mode = 2)

    ## Neighbor search.
    vec = find_neighbors(p, dist, nvec, mode = 2)

    return vec

def find_distances(p, mode):
    m    = len(p[:, 0])                                                             # The size if the triangulation is obtained.

    if mode == 1:
        dmin = np.zeros([m, 1]) + 1                                                 # dmin initialization with a "big" value.

        ## Delta computation.
        for i in np.arange(m):                                                      # For each of the nodes.
            x    = p[i, 0]                                                          # x coordinate of the central node.
            y    = p[i, 1]                                                          # y coordinate of the central node.
            for j in np.arange(m):                                                  # For all the nodes.
                if i != j:                                                          # If the the node is different to the central one.
                    x1 = p[j, 0]                                                    # x coordinate of the possible neighbor.
                    y1 = p[j, 1]                                                    # y coordinate of the possible neighbor.
                    d  = np.sqrt((x - x1)**2 + (y - y1)**2)                         # Distance from the possible neighbor to the central node.
                    dmin[i] = min(dmin[i],d)                                        # Look for the distance to the closest node.
        dist = (3/2)*np.max(dmin)                                                   # The distance is the maximum distance between two nodes.

    if mode == 2:
        ## Delta computation.
        p_expanded    = np.expand_dims(p, axis=1)                                   # Expand p to use vectorized operations.
        differences   = p_expanded - p                                              # The distance between each node and all the others is computed.
        distances     = np.sum(differences**2, axis=2)                              # The sum x^2 + y^2 is perform to compute the Euclidean norm.
        np.fill_diagonal(distances, np.inf)                                         # Distances to the self node are state as infinity and not zero.
        min_distances = np.sqrt(np.min(distances, axis=1))                          # Look for the distance to the closest node.
        dist          = (3/2)*np.max(min_distances)                                 # The distance is the maximum distance between two nodes.
    
    return dist

def find_neighbors(p, dist, nvec, mode):
    m    = len(p[:, 0])                                                             # The size if the triangulation is obtained.
    vec  = np.zeros([m, nvec], dtype = int) - 1                                     # The array for the neighbors is initialized.

    if mode == 1:
        for i in np.arange(m):                                                      # For each of the nodes.
            x     = p[i, 0]                                                         # x coordinate of the central node.
            y     = p[i, 1]                                                         # y coordinate of the central node.
            temp  = 0                                                               # Temporal variable as a counter.
            dst   = []
            index = []
            for j in np.arange(m):                                                  # For all the interior nodes.
                if i != j:                                                          # Check that we are not working with the central node.
                    x1 = p[j,0]                                                     # x coordinate of the possible neighbor.
                    y1 = p[j,1]                                                     # y coordinate of the possible neighbor.
                    d  = np.sqrt((x - x1)**2 + (y - y1)**2)                         # Distance from the possible neighbor to the central node.
                    if d < dist:                                                    # If the distance is smaller or equal to the tolerance distance.
                        dst.append(d)                                               # Save the distance to the neighbor node.
                        index.append(j)                                             # Store the neighbor node.
            
            index = [x for _, x in sorted(zip(dst, index))]                         # Sort the neighbors by distance.

            for idx, j in enumerate(index):                                         # For each stored neighbor.
                if idx < nvec:                                                      # If the number of neighbors is smaller than nvec.
                    vec[i, idx] = j                                                 # Store the neighbor.
    
    if mode == 2:
        dx = np.expand_dims(p[:,0], 1) - np.expand_dims(p[:,0], 0)                  # Compute dx between all the nodes.
        dy = np.expand_dims(p[:,1], 1) - np.expand_dims(p[:,1], 0)                  # Compute dy between all the nodes.
        
        radius = np.sqrt(dx**2 + dy**2)                                             # Get the distances from each node to all the others.
        
        for i in range(m):                                                          # For each node.
            neighbors = np.where((radius[i,:] < dist) & (np.arange(m) != i))[0]     # The neighbors are all the nodes within the radius.

            if len(neighbors) > 0:                                                  # If there are more neighbors than the requested.
                neighbors = neighbors[np.argsort(radius[i, neighbors])][:nvec]      # The neighbors are sorted by distance and only the closest nvec neighbors remains.
                vec[i, :len(neighbors)] = neighbors                                 # The nvec neighbors are stored.
            else:
                neighbors = neighbors[np.argsort(radius[i, neighbors])]             # The neighbors are sorted by distance and only the closest nvec neighbors remains.
                vec[i, :len(neighbors)] = neighbors                                 # The neighbors are stored.

    return vec