# Main demo script for running an experiment
import argparse
from skimage import data, segmentation

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import cv2

import os
import glob
import skimage.io as sio
import tqdm
import datetime
import time
import copy
import random

from skimage.segmentation import watershed, slic
from src.segment.watershed import watershed_segmentation
from src.segment.watershed import watershed_segmentation
from src.rag import RAG
from src.nng import NNG
from src.utils import visualize_rag, quantize_image, visualize_nng
from src.sprt import compute_averageColor, compute_conditionalProbability, compute_averageColor, sample_from_region, sprt_test
from src.region_merging import RegionMerging, NNGRegionMerging

def plot_segmentationLabels(img, labels):
	plt.figure(0, figsize=(10, 7))
	plt.subplot(1, 2, 1)
	plt.imshow(img)

	plt.subplot(1, 2, 2)
	plt.imshow(labels)

	plt.show()

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
	# path = '/home/saiamrit/Documents/BSR_bsds500/BSR/BSDS500/data/images/test/'
	path = args.input_path
	output_path = args.output
	
	# Load the image from the image_path parameter
	if(os.path.isdir(path)):
		files = glob.glob(path + '*.jpg')
		file = random.choice(files)

	name = file.split('/')[-1][:-4]
	print("Operating on file: ",name)

	# Create the output directory
	save_path = os.path.join(output_path , '{}'.format(name))
	os.mkdir(save_path)

	image = cv2.imread(file)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	w, h = int(image.shape[0]/2), int(image.shape[1]/2)
	image = cv2.resize(image, (h, w))

	# Compute the labels using either slic or watershed
	if(args.watershed):
		gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		labels = watershed_segmentation(gray_image)
	else:
		labels = slic(image, compactness=10, n_segments=300, start_label=1)
	
	# plot_segmentationLabels(image, labels)

	# Generate the NNG graph and start running the algorithm
	drm = NNGRegionMerging(image, labels, lambda1=0.95, lambda2=0.1)
	start = time.time()
	# start = datetime.datetime.now()
	initial_labels = drm.get_labels()
	initial_graph = copy.deepcopy(drm.graph)
	initial_nng_graph = copy.deepcopy(drm.nng)
	print("Number of initial regions: ", len(drm.graph.nodes))

	# Run the region merging process for max_iters number of times
	drm.run_region_merging(100)

	new_labels = drm.get_labels()
	print("Number of final regions: ", len(drm.graph.nodes))
	end = time.time()
	print('Segmentation takes: {:.4f} secs'.format((end-start)))

	# Save the outputs
	plt.imsave(save_path+'/Original Image.png',image)
	plt.imsave(save_path+'/Initial Labels.png',initial_labels)
	plt.imsave(save_path+'/Final labels.png',new_labels)
	# plot_segmentationLabels(initial_labels, new_labels)

	# Save the outputs for initial RAG and NNG, and final RAG and NNG.
	initial_label_img = visualize_rag(image, initial_labels, initial_graph, method = 'rag', time = 'initial', path = save_path)

	_ = visualize_nng(image, initial_labels, initial_nng_graph, method = 'nng', time = 'initial', path = save_path)

	final_label_image = visualize_rag(image, new_labels, drm.graph, method = 'rag', time = 'final', path = save_path)

	_ = visualize_nng(image, new_labels, drm.nng, method = 'nng', time = 'final', path = save_path)

	plt.imsave(save_path+'/Initial Label Image.png',initial_label_img)
	plt.imsave(save_path+'/Final label Image.png',final_label_image)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Segmentation parameters Parser. Pass the parameters following instructions given\
	below to run the demo experiment.")
	# Input and output paths
	parser.add_argument('--input_path', default="./img/", type=str, help='Path to image for processing')
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
	