"""
Functions for calculating Mahalanobis related functions.

'eigenDecomp' produces the invcovariance matrix and eigenvectors
necessary for projecting measurements to class subspaces and cacluating
their distances from mean.

'calcMahalDistance' returns the mahalanobis distance of a reduced measurement.

'calcMahalProb' calculates the likelihood of chance occurence of a given measurement  # noqa: E501
within a set of mahalanobis distances.
"""
import bisect
import numpy as np
from typing import List, Tuple, Any


def eigenDecomp(matrix: List[List[float]], threshold: float) -> List[Any]:
	"""
	Performs principal components analysis on matrix and returns
	necessary vectors to project matrix while preserving threshold
	value of variance. Also provides the inverse covariance matrix
	of new projected values so as to calculate mahalanobis distance
	for new measures.

	For cases where there is no variance (and thus all measures for
	a class are the same), the function returns original measurements
	as eigenvectors and a 0 matrix for invCovariance. This is used
	for other functions to assume these measurements as distinguishing
	features. (Note, an inverse covariance of 0 would only naturally
	occur if there was infinite variance, which should not occur in
	normal computations. Thus we avoid overlap.)

	Inputs:

		matrix: Numpy matrix of measurement tuples. Each row is a measurement
		each column is a variable. Assumes measurements have been processed
		and centered around mean.

		theshold: float specifying threshold of variance PCA must preserve.

	Outpus:

		invCovariance: inverse covariance matrix. Is 0 vector if there is
		no variance in measurement set.

		eigenVectors: eigenvectors that span subspace that accomodates theshold
		percentage of variance in data. Used to project measurements to reduced
		space.
	"""

	# Calculates covariance matrix. Numpy gets a bit funny with 1x1 cases so
	# we're going to force matrix instead of array.
	covarMatrix = np.asmatrix(np.cov(matrix, rowvar=False, bias=False))  # Bias is labeled for troubleshooting.  # noqa: E501

	# If there's no variance, we assume this feature to explicitly categorize the
	# space.
	if np.all((covarMatrix == 0)):
		# We'll return none and the matrix as an indicator that this needs
		# to be processed differently.
		return None, matrix

	# Perform eigen decomposition. Covariance is postive singular.
	eigenValues, eigenVectors = np.linalg.eigh(covarMatrix)

	# Numpy orders ascending. So we need to flip them

	# Find reversed index order.
	reverse = np.flip(eigenValues.argsort())

	# Apply to vectors and values.
	eigenValuesSorted = eigenValues[reverse]
	eigenVectorsSorted = eigenVectors[:, reverse]

	# Identifies minimum span of eigenvectors by examining proportion of
	# eigenvalue sums. Stops when threshold is met (or all vectors used).
	index = 1
	total = sum(eigenValuesSorted)
	currentSum = eigenValuesSorted[0]
	while currentSum/total < threshold and index < len(eigenValuesSorted):
		currentSum += eigenValuesSorted[index]
		index += 1

	# Makes diagonal of eigenvalues (i.e. covariance matrix of new measure).
	covar = np.asmatrix(np.diag(eigenValuesSorted[:index]))

	# and inverts
	invCovar = np.linalg.inv(covar)

	return invCovar, eigenVectorsSorted[:, :index]


def calcMahalDistance(measure: Tuple[float], invCovar: List[List[float]],
	eigenVectors: List[List[float]]) -> float:
	""" Calculates mahalanobis distance for measure after projection
	into subspace defined by eigenVectors.

	Inputs:

		measure: Tuple of floats representing a single measurement.

		invCovar: matrix of the inverse-covariance of the target space.

		eigenVectors: matrix of Principal Components used for projecting
		to target space.

	Outpus:

		distance: float of the squared mahalanobis distance.
	"""

	# Checks of invCovariance is None (which means there was no variance in training)  # noqa: E501
	if invCovar is None:
		# Returns a None distance as placeholder. (This is only checked by the tagger, which is already dealing with this.)  # noqa: E501
		return None

	# Need to transpose measures (since notation has them as column vectors, 
	# not row vectors as we do.) Project to space.
	y = np.transpose(eigenVectors) @ np.transpose(measure)

	# Calculate mahal distance.
	yT = np.transpose(y)
	dist = yT @ invCovar @ y

	return dist

def calcMahalProb(val: float, distances: List[float]) -> float:
	"""
	Calculates probability that given value would occur by chance
	in the space defined by distance distribution. Probabilities
	are calculated by assuming each value of distance quantizes
	a similar probability space, adjusted by .5 to avoid overlapping
	distributions.

	Inputs:

		val: float value to be considered.

		distances: ordered array of mahal distance distributions
		for given space.

	Outpus:

		prob: float determining likilihood such value occurs by chance.
	"""

	# If larger than all distances.
	if val > distances[-1]:
		# 100% chance.
		return 1

	# If smaller than all other distances.
	elif val < distances[0]:
		# Falls into smallest quantization.
		return 1 / len(distances)

	else:
		# We locate the quantization that makes most sense.
		index = bisect.bisect_right(distances, val)
		return (index + .5) / len(distances)  # + 1 for indexing, -.5. See notes.

def main():
	# Test eigendecomp to make sure values and vectors are accurately computed taken from: https://towardsdatascience.com/the-mathematics-behind-principal-component-analysis-fff2d7f4b643  # noqa: E501
	# Note, this person doesn't unbias the data.Also, they don't standardize the vectors.  # noqa: E501
	# Need to multiply their results by value corresponding to their 1.
	matrix = [[90, 60, 90], [90, 90, 30], [60, 60, 60], [60, 60, 90], [30, 30, 30]]  # noqa: E501
	matrix = np.asmatrix(matrix)
	print(eigenDecomp(matrix, .95))

	print()

	# Test eigendecomp to make sure estimation is accurate. https://iksinc.online/2018/08/21/principal-component-analysis-pca-explained-with-examples/
	matrix = [[7, 4, 3], [4, 1, 8], [6, 3, 5], [8, 6, 1], [8, 5, 7], [7, 2, 9], [5, 3, 3], [9, 5, 8], [7, 4, 5], [8, 2, 2]]
	matrix = np.asmatrix(matrix)
	print(eigenDecomp(matrix, .650)) # Eigen values total 12.7. .94 should yield two vectors. .95 yields three. .65 yields one.

	print()

	# Test mahal distance
	measure = [1, 2, 3]
	invCovar = [[1, 0], [0, 1]]
	vectors = [[0, 0], [0, 1], [1, 0]]
	print(calcMahalDistance(measure, invCovar, vectors)) # Should be 13.

	print()

	# Test calc mahal probability
	distances = [1, 2, 3, 4, 5]
	print(calcMahalProb(.5, distances))  # Should be .1
if __name__ == '__main__':
	main()
