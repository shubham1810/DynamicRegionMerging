import os

import numpy as np
import tqdm
import copy


class NNG(object):
    def __init__(self, graph):
        """
        NNG Class takes a RAG as graph input and constructs the
        Nearest Neighbour Graph from it.
        """
        self.graph = graph
        self.nodes = copy.deepcopy(self.graph.nodes)
        self.edges = {}
        self.edge_data = {}
        
        # Update the directed graph
        self.make_min_graph()
    
    def make_min_graph(self):
        for nx in self.nodes:
            neighbours = self.graph.edge_data[nx]
            dist = []
            
            for neigh_x in neighbours:
                dist.append(self.graph.edges[(nx, neigh_x)])
            
            idx_min = np.argmin(dist)
            neigh_min = neighbours[idx_min]
            self.edge_data[nx] = neigh_min
            self.edges[(nx, neigh_min)] = self.graph.edges[(nx, neigh_min)]
    
    def find_cycles(self):
        potential_merges = []
        
        for ix in self.edge_data:
            compare = self.edge_data[ix]
            match = self.edge_data[compare]

            if (ix == match) and (ix < compare):
                potential_merges.append((ix, compare))

        return potential_merges