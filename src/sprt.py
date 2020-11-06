import numpy as np


def sample_from_region(region, n_points):
	"""
	Takes a region and number of points to be sampled, 
	returns a sampled number of points from the region.
	"""
	ind = np.random.permutation(range(region.shape[0]))[:n_points]
	return region[ind,:3]

def compute_averageColor(sample):
	"""
	Takes a sample region and returns the average color 
	of all the pixels belonging to the sample region
	"""
	mean = sample.mean(axis=0).reshape((1,sample.shape[1]))
	return mean

def compute_regionCovariance(sample1, sample2):
	"""
	Takes 2 sample regions and returns the covariance
	of the stacked region over their feature dimension.
	"""
	m = np.vstack((sample1,sample2))
	cov = np.cov(m.T)
	return cov

def compute_conditionalProbability(region1, region2, lambda1, lambda2):
	"""
	Takes 2 regions and computes the following:
	1. Samples half of the minimum no of pixels among both the regions.
	2. Create a new region by taking a UNION of the 2 regions and
		samples an equal number of pixels from the new region.
	3. Computes the average color of the 3 sampled regions.
	4. Computes the covariance of the sample regions across color dimensions
	5. Computes the conditional probability to estimate the distribution of
		color cues by fitting a gaussian distribution.

	Returns the computed conditional probability distributions.
	"""

	# The minimum no of pixels among both regions.
	l = min(region1.shape[0]//2,region2.shape[0]//2)

	# Sample from the regions
	sample1 = sample_from_region(region1, l) 
	sample2 = sample_from_region(region2, l)

	# Create new region as concatination of the 2 regions and sample from it.
	region3 = np.concatenate((region1, region2), axis=0)
	sample3 = sample_from_region(region3, l)
	
	# Compute average color of all regions.
	s1_avg = compute_averageColor(sample1)
	s2_avg = compute_averageColor(sample2)
	s3_avg = compute_averageColor(sample3)
	
	# Compute the covariance of regions 
	cov = compute_regionCovariance(sample1,sample2)

	"""
	 Compute the exponential coefficients.
	 Some covariance matrices are singular matrices so inplace 
	 of inverse, a pseudo inverse is computed.
	"""
	ex1 = np.exp(np.dot(-(s2_avg - s1_avg), np.dot(np.linalg.pinv(cov),
												(s2_avg - s1_avg).T)))
	ex2 = np.exp(np.dot(-(s2_avg - s3_avg), np.dot(np.linalg.pinv(cov),
												(s2_avg - s3_avg).T)))

	# Compute the conditional probabilities
	p_h1 = 1 - (lambda1 * ex1)
	p_h2 = 1 - (lambda2 * ex2)

	return p_h1, p_h2


def SPRT(nodes, edge_data, alpha=0.05, beta=0.05, h0=0, h1=1):
	"""
	Main function that performs the SPRT test between all the nodes in the graph
	and their neighbour nodes.
	"""

	"""
	Fixing essential parameters values. The lambda values are the scaling factor
	for the exponential coefficients in the conditional probability estimation
	of the cues and the neta values aree used to compute the N value, the upper 
	limit on the no of tests.
	"""
	lambda1 = 1
	lambda2 = 1
	nita0 = 0.1
	nita1 = 0.1

	"""Computing the other necessary parameters. The A and B values are the upper 
	and lower limits of the range in which delts falls.
	""" 
	A = np.log((1 - beta)/alpha)
	B = np.log(beta/(1 - alpha))

	result = [] # stores the final results.


	# The Expectations, the max of which serves as the upper limit for no of tests
	E1 = ((A * alpha) + (B * (1-alpha)))/nita0
	E2 = ((A * (1 - beta)) + (B * beta))/nita1
	N = max(E1, E2)
	result = []

	a,b,c,d = 0, 0, 0, 0

	# Iterate over all the nodes.
	for ix in sorted(nodes.keys()):
		for iy in edge_data[ix]:
			
			# Not iterating over the same pair again.
			if iy > ix:

				# initializing the delta and counter to 0
				delta,n = 0, 0 
				decision = ''

				# iterating till delta lies in the range [B <= delta <= A]
				while(delta >= B and delta <= A): 

					# rejecting segments which have less than 2 pixels.
					if(len(nodes[ix])>=2 and len(nodes[iy])>=2): 

						p_h1, p_h2 = compute_conditionalProbability(nodes[ix], nodes[iy], lambda1, lambda2)

						# Update the evidance accumulator delta with the likelihood ratio
						delta = delta + np.log((p_h1 * (1 - beta))/(p_h2 * (1 - alpha)))

						# Update the trial counter
						n += 1

						print(n)
						print(B, delta[0][0], A)

						# Condition checks
						if(n < N):
						
							if(delta >= A):
								decision = "consistent"
								a += 1
								result.append([ix,iy,decision])
								break

							elif(delta <= B):
								decision = "inconsistent"
								b += 1
								result.append([ix,iy,decision])
								break

						elif(n > N):
							if(delta >= 0):
								decision = "consistent"
								c += 1
								result.append([ix,iy,decision])
								break

							elif(delta < 0):
								decision = "inconsistent"
								d += 1
								result.append([ix,iy,decision])
								break

	print(a,b,c,d)

	return result


