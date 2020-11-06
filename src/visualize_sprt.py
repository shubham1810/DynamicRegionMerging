import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

def visualize_sprt(img, labels, graph, sprt):

	plt.figure(0, figsize=(20, 10))
	
	plt.subplot(1, 2, 1)

	# Plot the segemtation boundary image first
	plt.imshow(mark_boundaries(img, labels))

	for ix in range(1, max(graph.edge_data)+1):
		curr = ix
		for iy in graph.edge_data[ix]:
			if iy > ix:
				region1_center = graph.nodes[ix][:, -2:].mean(0)
				region2_center = graph.nodes[iy][:, -2:].mean(0)

				plt.text(region1_center[1], region1_center[0], str(ix), fontsize=15)
				# plt.text(region2_center[0], region2_center[1], str(iy), fontsize=15)

	im = img.copy()

	plt.subplot(1, 2, 2)
	# plt.imshow(labels)
	for s in sprt:
		if(s[2] == 'consistent'):
			# a,b,c = 0,255,0
			a,b,c = random.randint(0,255), random.randint(0,255), random.randint(0,255)
			im[graph.nodes[s[0]][:,3],graph.nodes[s[0]][:,4]] = [a,b,c]
			im[graph.nodes[s[1]][:,3],graph.nodes[s[1]][:,4]] = [a,b,c]
	plt.imshow(im)


	plt.show()


