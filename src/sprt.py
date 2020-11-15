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
    mean = sample.mean(axis=0)
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
    Ia = compute_averageColor(sample1)
    Ib = compute_averageColor(sample2)
    Iab = compute_averageColor(sample3)
    
    # Compute the covariance of regions 
    S = compute_regionCovariance(sample1,sample2)

    """
     Compute the exponential coefficients.
     Some covariance matrices are singular matrices so inplace 
     of inverse, a pseudo inverse is computed.
    """
    p_h0 = 1 - (lambda1 * np.exp(- (Ib - Ia) @ np.linalg.pinv(S) @ (Ib - Ia).T))
    p_h1 = 1 - (lambda2 * np.exp(- (Ib - Iab) @ np.linalg.pinv(S) @ (Ib - Iab).T))
    
    return p_h0, p_h1


def sprt_test(node1, node2):
    """
    Take data from two nodes, and merge them if they pass the SPRT test.
    
    Args:
        node1: points in the first node
        node2: ponits in the second node
    """
    merge = False
    
    """
    Fixing essential parameters values. The lambda values are the scaling factor
    for the exponential coefficients in the conditional probability estimation
    of the cues and the neta values aree used to compute the N value, the upper 
    limit on the no of tests.
    """
    alpha=0.05
    beta=0.05
    h0=0
    h1=1
    lambda1 = 1
    lambda2 = 1
    eta0 = 0.1
    eta1 = 0.1

    """Computing the other necessary parameters. The A and B values are the upper 
    and lower limits of the range in which delts falls.
    """ 
    A = np.log((1 - beta)/alpha)
    B = np.log(beta/(1 - alpha))

    # The Expectations, the max of which serves as the upper limit for no of tests
    E0 = ((A * alpha) + (B * (1-alpha)))/eta0
    E1 = ((A * (1 - beta)) + (B * beta))/eta1
    N = max(E0, E1)

    # Placeholder storage for results
    result = []

    # Counters to check which condition is hit how many times, printing the values at the end
    a,b,c,d = 0, 0, 0, 0

    # Initialize delta and counter values
    delta = 0.0
    n = 0
    
    # Check if number of pixels are enough or not
    if node1.shape[0] <= 2 or node2.shape[0] <= 2:
        return merge
    
    # Begin the SPRT test (sample data and update delta sequentially)
    while delta >= B and delta <= A:
        # Compute the null and alternate hypothesis values
        p_h0, p_h1 = compute_conditionalProbability(node1, node2, lambda1, lambda2)
        # print(p_h0, p_h1, delta)
        # Update the counter for number of trials
        n += 1
        
        # Update delta
        delta = delta + np.log((p_h1 * (1 - beta))/(p_h0 * (1 - alpha)))
        
        # Check number of steps
        if n > N:
            break
    
    # Make a decision for merging
    if n <= N:
        if delta >= A:
            # print("A")
            return True
        else:
            # print("B")
            return False
    else:
        if delta >= 0:
            # print("C")
            return True
        else:
            # print("D")
            return False