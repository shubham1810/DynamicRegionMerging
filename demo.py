# Main demo script for running an experiment
import argparse
from skimage import data, segmentation

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import cv2

import os
import skimage.io as sio
import tqdm
import datetime
import copy

from src.segment.watershed import watershed_segmentation
from src.rag import RAG
from src.nng import NNG
from src.utils import visualize_rag, quantize_image, visualize_nng
from src.sprt import compute_averageColor, compute_conditionalProbability, compute_averageColor, sample_from_region, sprt_test
from src.region_merging import RegionMerging, NNGRegionMerging


def main(args):
    """
    Main function takes the arguments and does the following:
        1. Load the image from the image_path parameter
        2. Compute the labels using either slic or watershed
        3. Generated the NNG graph and start running the algorithm
        4. Run the region merging process for max_iters number of times
        5. Visualize the outputs if we want
        6. Create the output directory and store the outputs
        7. Save the outputs for initial RAG and NNG, and final RAG and NNG.
    """
    print(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segmentation parameters Parser. Pass the parameters following instructions given\
    below to run the demo experiment.")
    # Input and output paths
    parser.add_argument('--image_path', default="./img/demo.png", type=str, help='Path to image for processing')
    parser.add_argument('--output', default="./output/", type=str, help="Path to folder for storing output")
    
    # RAG/NNG parameters
    parser.add_argument('--watershed', default=False, action='store_true', help="Whether to use watershed for initial segmentation")
    parser.add_argument('--lambda1', default=0.8, type=float, help="Value for Lambda1 parameter")
    parser.add_argument('--lambda2', default=0.1, type=float, help="Value for Lambda2 parameter")
    parser.add_argument('--alpha', default=0.05, type=float, help="Parameter value for alpha")
    parser.add_argument('--beta', default=0.05, type=float, help="Parameter value for beta")
    parser.add_argument('--visualize', default=False, action='store_true', help='Visualize the outputs of the algorithm')
    parser.add_argument('--max_iters', default=100, type=int, help='Max iterations for the Region merging loop')

    
    # Parse the arguments
    args = parser.parse_args()

    # Call the main function
    main(args)
    