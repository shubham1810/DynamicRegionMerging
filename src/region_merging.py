import numpy as np
from .rag import RAG
from .nng import NNG
from .sprt import *
from skimage import data, segmentation
import datetime
import cv2
import tqdm


class RegionMerging:
    def __init__(self, img, labels, alpha=0.05, beta=0.05, lambda1=1.0, lambda2=0.5):
        # we keep a copy of the image and the initial labels
        self.img = img.copy()
        self.labels = labels.copy()
        
        # Save parameters
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        # Placeholders for the graph
        self.graph = None
        
        # Initialize the first graph
        self.make_rag(timed=True)
    
    def run_region_merging(self, max_iters=1):
        """
        Run the region merging process for a maximum of
        max_iters, or until the segmented image stays the same.
        """
        
        for iteration in tqdm.tqdm(range(max_iters)):
            # Compute the merged regions
            self.run_merging_pass()
            
            # Update the RAG
            self.make_rag()
        
    
    def make_rag(self, timed=False):
        """
        Build the RAG based on the current image and the label map
        """
        # Set timed to True if we want to see how long it takes to
        # build the graph.
        if timed:
            start = datetime.datetime.now()
        self.graph = RAG(self.img, self.labels)
        
        if timed:
            print(datetime.datetime.now() - start)
    
    def check_region_consistency(self, candidates):
        """
        Check for SPRT test for the potential candidates
        for region merging and return a list of whether we can merge the
        corresponding regions or not.
        """
        results = []
        for cx in candidates:
            test_val = sprt_test(self.graph.nodes[cx[0]], self.graph.nodes[cx[1]], alpha=self.alpha, 
                                 beta=self.beta, lambda1=self.lambda1, lambda2=self.lambda2)
            results.append([cx[0], cx[1], test_val])
        return results
    
    def check_merging_predicate(self):
        """
        Check the minimum edge between nodes and keep the ones which are min
        to each other exclusively.
        """
        # Placeholder for min map
        min_map = {}
        
        # For each node in the graph, check the data
        for nx in self.graph.nodes:
            # List of nodes connected to current
            neighbours = self.graph.edge_data[nx]
            dist = []
            
            for neigh_x in neighbours:
                dist.append(self.graph.edges[(nx, neigh_x)])
            
            # get the minimum distant node from the current one
            idx_min = np.argmin(dist)
            neigh_min = neighbours[idx_min]
            # Update the node id mappings
            min_map[nx] = neigh_min
        
        # Keep a list of potential merges
        potential_merges = []
        
        for ix in min_map:
            compare = min_map[ix]
            match = min_map[compare]
            
            if (ix == match) and (ix < compare):
                potential_merges.append((ix, compare))
        
        return potential_merges
    
    def run_merging_pass(self, verbose=False):
        merge_candidates = self.check_merging_predicate()
        
        if verbose:
            print("Merging candidates: ", merge_candidates)
        
        merges = self.check_region_consistency(merge_candidates)
        
        if verbose:
            print("Final Merges: ", merges)
        
        # Apply merging by updating the label map
        lbs = self.labels.copy()
        
        for res in merges:
            reg1, reg2, can_merge = res
            if can_merge:
                lbs[lbs == reg2] = reg1
        
        self.labels = lbs.copy()
        
    def get_labels(self):
        return self.labels
    
    def get_quant_image(self):
        """
        Get a pseudo-quantized version of the image
        """
        new_img = np.zeros_like(self.img)
        for lx in np.unique(self.labels):
            mask = (self.labels == lx)[:, :, None]
            vals = mask*self.img
            col = vals.reshape((-1, 3)).sum(0) / mask.reshape((-1,)).sum(0)
            new_img += (mask*col).astype(np.uint8)
        
        return new_img


class NNGRegionMerging:
    def __init__(self, img, labels, alpha=0.05, beta=0.05, lambda1=1.0, lambda2=0.5):
        # we keep a copy of the image and the initial labels
        self.img = img.copy()
        self.labels = labels.copy()
        
        # Save parameters
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        # Placeholders for the graph
        self.graph = None
        self.nng = None
        
        # Initialize the first graph
        self.make_rag(timed=True)
    
    def run_region_merging(self, max_iters=1):
        """
        Run the region merging process for a maximum of
        max_iters, or until the segmented image stays the same.
        """
        
        for iteration in tqdm.tqdm(range(max_iters)):
            # Compute the merged regions
            self.run_merging_pass()
            
            # Update the RAG
            self.make_rag()
        
    
    def make_rag(self, timed=False):
        """
        Build the RAG based on the current image and the label map
        """
        # Set timed to True if we want to see how long it takes to
        # build the graph.
        if timed:
            start = datetime.datetime.now()
        self.graph = RAG(self.img, self.labels)
        self.nng = NNG(self.graph)
        
        if timed:
            print(datetime.datetime.now() - start)
    
    def check_region_consistency(self, candidates):
        """
        Check for SPRT test for the potential candidates
        for region merging and return a list of whether we can merge the
        corresponding regions or not.
        """
        results = []
        for cx in candidates:
            test_val = sprt_test(self.nng.nodes[cx[0]], self.nng.nodes[cx[1]], alpha=self.alpha, 
                                 beta=self.beta, lambda1=self.lambda1, lambda2=self.lambda2)
            results.append([cx[0], cx[1], test_val])
        return results
    
    def run_merging_pass(self, verbose=False):
        merge_candidates = self.nng.find_cycles()
        
        if verbose:
            print("Merging candidates: ", merge_candidates)
        
        merges = self.check_region_consistency(merge_candidates)
        
        if verbose:
            print("Final Merges: ", merges)
        
        # Apply merging by updating the label map
        lbs = self.labels.copy()
        
        for res in merges:
            reg1, reg2, can_merge = res
            if can_merge:
                lbs[lbs == reg2] = reg1
        
        self.labels = lbs.copy()
        
    def get_labels(self):
        return self.labels
    
    def get_quant_image(self):
        """
        Get a pseudo-quantized version of the image
        """
        new_img = np.zeros_like(self.img)
        for lx in np.unique(self.labels):
            mask = (self.labels == lx)[:, :, None]
            vals = mask*self.img
            col = vals.reshape((-1, 3)).sum(0) / mask.reshape((-1,)).sum(0)
            new_img += (mask*col).astype(np.uint8)
        
        return new_img