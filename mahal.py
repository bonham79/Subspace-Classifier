"""
Functions for mahalanobis calculations.
"""
import numpy as np
from typing import List, Tuple


def eigenDecomp(matrix, threshold):
	# Takes a matrix and performs an
	# eigenvalue decomposition to find minimum eigenvectors
	# needed to exceed minimal percentage of eigenvalue proportion.
	# Returns inverse covariance matrix with diagonals of eigenvalues and
	# subset of eigenvectors as a projection array.

	# Calculates covariance matrix
	covarMatrix = np.asmatrix(np.cov(matrix, rowvar=False, bias=False))
	print(covarMatrix)

	# Perform eigen decomposition
	eigenValues, eigenVectors = np.linalg.eig(covarMatrix)

	# Sorts values since numpy doesn't necessarily order.
	eigenValuesSorted = np.flip(np.sort(eigenValues))
	eigenVectorsSorted = eigenVectors[:, np.flip(eigenValues.argsort())]  # I hate numpy sorting...  # noqa: E501
	# Identifies minimum span of eigenvectors by examining proportion of
	# eigenvalue sums.
	total = sum(eigenValuesSorted)
	index = 1
	currentSum = eigenValuesSorted[0]
	while currentSum/total < threshold and index <= len(eigenValuesSorted):
		index += 1
		currentSum = sum(eigenValuesSorted[:index])
	
	# Makes diagonal of eigenvalues (i.e. covariance matrix of new measure)
	covar = np.asarray(np.diag(eigenValuesSorted[:index]))

	# and inverts
	invCovar = np.linalg.inv(covar)

	return invCovar, eigenVectorsSorted[:, :index]


def calcMahalDistance(measure, invCovar, eigenVectors):
	# Calculates the projected Mahalanobis distance for a single measure.

	measure = np.asmatrix(measure)
	invCovar = np.asmatrix(invCovar)
	eigenVectors = np.asmatrix(eigenVectors)
	y = np.transpose(eigenVectors) @ np.transpose(measure)  # measures are traditionally column vectors. we've been treating them as row vectors. Oops  # noqa: E501
	yT = np.transpose(y)
	dist = yT @ invCovar @ y
	return dist

def calcMahalProb(val, distances):
	# Calculates probability of meas not being of class
	# spanned by distances. assumes distances are sorted.
	
	
	if val < distances[0]:  # All values are greater.
		return 1/len(distances)
	elif val > distances[-1]:  # There is no greater value..
		return 1
	else:
		index = binCompare(distances, val) + 1 # Notes are for fortran indexing
		return (index - .5)/len(distances)

def binCompare(arr: Tuple[float], val: float)-> int:
	# Binary search for comparing val x in array arr. Finds index of first value greater than x.  # noqa: E501
	start = 0
	end = len(arr) - 1
	index = -1
	while start <= end:
		mid = (start + end) // 2

# Move to right side if target is
# greater.
		if arr[mid] <= val:
			start = mid + 1

       # Move left side.
		else:
			index = mid
			end = mid - 1
	return index

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
	print(calcMahalProb(0, distances))  # Should be .3
if __name__ == '__main__':
	main()