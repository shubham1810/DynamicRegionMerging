import os

import numpy as np
import tqdm


class RAG(object):
    """
    Region Adjacency Graph

    Using a provided image and corresponding label map, create the RAG for connected regions
    based on the color properties of the enclosed vertices (pixels) for different regions.
    """
    def __init__(self, img, labels):
        # Define some fixed parameters here
        self.padding = 1
        
        # placeholder for nodes and edges
        self.nodes = {}
        self.edges = {}
        
        # Add padding to the label image
        labels_pad = np.pad(labels, self.padding, mode='edge')
        
        # [TODO]: Generalize to 4-neighbours or 8-neighbours using a structuring kernel.
        # Check 8-connected neighbours for regions
        for ix in np.ndindex(labels.shape):
            loc_u, loc_v = ix[0] + self.padding, ix[1] + self.padding
            self.check_region_edge(labels_pad[loc_u - self.padding:loc_u + self.padding + 1, 
                                              loc_v - self.padding:loc_v + self.padding + 1])
            
        # Iterate over each pixel in the image
        for ix in np.ndindex(labels.shape):
            # Image color-position value
            col_pos_val = np.concatenate([img[ix], ix])
            
            # node value
            node_val = labels[ix]
            
            # Update the nodes of the graph
            # [NOTE]: the vector saved for each pixel is 5D, signifying the following values: 
            # (r, g, b, x, y), where (x, y) is the position of the pixel.
            self.nodes[node_val].append(col_pos_val)
        
        # Update each node and replace list with numpy array (faster processing for later)
        for ndx in self.nodes:
            self.nodes[ndx] = np.array(self.nodes[ndx])
    
    def check_region_edge(self, values):
        """
        Given a list of values, find the center and compare with the neighbours.
        If center is connected to a different region, add an edge in the graph.
        """
        values = values.flatten().astype(np.int)
        center = len(values)//2
        
        for vx in range(len(values)):
            if not (vx == center):
                # Add an edge if it doesn't exist already
                self.add_edge(values[vx], values[center])
    
    def add_edge(self, u, v):
        """
        Method to add the edge (u, v) with empty data {}.
        Return Nothing if edge already exists.
        """
        # Check if the edge already exists
        if (u,v) in self.edges:
            return
        
        # If the nodes don't exist then add them
        self.add_node(u)
        self.add_node(v)
        
        # Add the edge
        self.edges[(u,v)] = {}
    
    def add_node(self, u):
        """
        This method adds a node to the RAG if it didn't exist already.
        Method does not return anything.
        Args:
            u: node id (int)
        """
        self.nodes.setdefault(u, [])