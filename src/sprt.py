import numpy as np

class SPRT(object):
	def __init__(self, nodes, alpha=0.05, beta=0.05, h0=0, h1=1):

		self.nodes = nodes
		self.lambda1 = 1
		self.lambda2 = 1
		self.alpha = alpha
		self.beta = beta
		self.h0 = h0
		self.h1 = h1
		self.N = 2
		# Necessary arguments
		self.A = np.log((1 - self.beta)/self.alpha)
		self.B = np.log(self.beta/(1 - self.alpha))
		self.decision = ''
		self.result = []

		#Calling test function
		self.SPRT_test()

	def sample_from_region(self,region):
		ind = np.random.choice(range(len(region)),(len(region)//2))
		reg = []
		for i in ind:
			reg.append(region[i][:3])
		return np.array(reg)


	def compute_averageColor(self,sample):
		mean_img = sample.mean(axis=1).reshape((sample.shape[0],1))
		return mean_img
 
	def compute_regionCovariance(self,sample1, sample2):
		
		m = np.hstack((sample1,sample2))
		cov = np.cov(m)
		return cov

	def compute_conditionalProbability(self, region1, region2):
		sample1 = self.sample_from_region(region1) 
		sample2 = self.sample_from_region(region2)

		region3 = np.concatenate((region1, region2), axis=0)
		sample3 = self.sample_from_region(region3)


		l = max(sample1.shape[0],sample2.shape[0])
	
		if(sample1.shape[0] != l):
			pad = np.zeros(((l - sample1.shape[0]),3))
			sample1 = np.concatenate((sample1, pad), axis=0)

		elif(sample2.shape[0] != l):
			pad = np.zeros(((l - sample2.shape[0]),3))
			sample2 = np.concatenate((sample2, pad), axis=0)

		# elif(sample3.shape[0] != l):
		# 	pad = np.zeros(((l - sample3.shape[0]),3))
		sample3 = sample3[:l]
		
		s1_avg = self.compute_averageColor(sample1)
		s2_avg = self.compute_averageColor(sample2)
		s3_avg = self.compute_averageColor(sample3)
		
		cov = self.compute_regionCovariance(sample1,sample2)


		ex1 = np.exp(np.dot(-(s2_avg - s1_avg).T, np.dot(np.linalg.pinv(cov), (s2_avg - s1_avg))))
		ex2 = np.exp(np.dot(-(s2_avg - s3_avg).T, np.dot(np.linalg.pinv(cov), (s2_avg - s3_avg))))

		p_h1 = 1 - (self.lambda1 * ex1)
		p_h2 = 1 - (self.lambda2 * ex2)

		return p_h1, p_h2

	def calc_Expectation(a, n):
		pass

		# prb = 1 / n 
		  
		# s = 0
		# for i in range(0, n): 
		#     s += (a[i] * prb)  
			  
		# return float(sum) 


	def SPRT_test(self):

		keys = sorted(list(self.nodes.keys()))

		for ix in range(1,len(keys)+1):
			for iy in range(ix+1,len(keys)+1):
				# if(self.nodes[ix] == self.nodes[iy]):
				# 	continue

				# else:
				# print(ix, iy)
				delta,n = 0, 0
				delta_ar = []

				# print(delta, self.A, self.B)

				while(delta >= self.B and delta <= self.A):
					# print(delta)

					p_h1, p_h2 = self.compute_conditionalProbability(self.nodes[ix],self.nodes[iy])

					delta = delta + np.log((p_h1*(1-self.beta))/(p_h2*(1-self.alpha)))
					delta_ar.append(delta)

					n += 1

					if(n < self.N):
					
						if(delta >= self.A):
							self.decision = "consistent"

						if(delta <= self.B):
							self.decision = "inconsistent"

					if(n > self.N):
						if(delta >= 0):
							self.decision = "consistent"

						if(delta < 0):
							self.decision = "inconsistent"

				print(ix,iy,self.decision)
				self.result.append([ix,iy,self.decision])


