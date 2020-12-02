"""
Module for building subspace classifier.
"""
import mahal
import feature
import numpy as np
from typing import Dict, List


class Classifier:
	# Classifier for tagging values. Built on class sorted dictionary of
	# mahalanobis distances and class sorted dict of eigenVectors for
	# projection.
	def __init__(self, featureTable: Dict['str', List[int]], mahalDistances: Dict['str', List[float]], invCovariance: Dict, eigenVectors: Dict) -> List[float]:  # noqa: E501
		self.distances = mahalDistances
		self.vectors = eigenVectors
		self.invCovariance = invCovariance
		self.features = featureTable

	def tag(self, meas, classThreshold):
		# Initializes look-up table Sets default of reserve for possibility no class
		# beats threshold.
		minimum = classThreshold
		tag = "RESERVE"

		for classValue in self.features:
			vect = self.vectors[classValue]
			invCov = self.invCovariance[classValue]

			# Narrows down measurements to essential features.
			reducedMeasures = [meas[feature] for feature in self.features[classValue]]
	
			# Converts for numpy.
			measArray = np.asarray(reducedMeasures)

			# Calculate mahalanobis distance
			dist = mahal.calcMahalDistance(measArray, invCov, vect)

			# Find likliehood of null-hypothesis for this class.
			py = mahal.calcMahalProb(dist, self.distances[classValue])

			# Checks if it is minimum.
			if py < minimum:
				tag = classValue
		return tag


def buildClassifier(measures: List[float], tags: List['str'], featureTable: Dict['str', List[int]],  eigenThreshold: float) -> Classifier:  # noqa: E501
	"""
	Function for processing data and building mahalanobis
	classifier. Identifies principle components for
	subspace of each class and then constructs classifier
	that can project new values to estimate class.

	Inputs:
	reducedMeasures: dict of class measurement pairs. Measurements
	are reduced to essential features for respective class.

	eigenThreshold: float indicate fraction of subspace to be
	spanned by eigen vectors during decomposition.
	"""

	# Groups measures by classes.
	reducedMeasures = feature.reduceFeatures(measures, tags, featureTable)
	mahalDistances = {val: [] for val in reducedMeasures}
	eigenVectors = {val: None for val in reducedMeasures}
	invCovariance = {val: None for val in reducedMeasures}

	# For each class.
	for tag in reducedMeasures:

		# Chooses measurements
		classMeasures = reducedMeasures[tag]
		
		# Converts to array for numpy use
		print(tag)
		classMatrix = np.asarray(classMeasures)
		print(classMatrix)
		# Identifies essential eigen vectors and values for class.
		invCov, eVec = mahal.eigenDecomp(classMatrix, eigenThreshold)
		eigenVectors[tag] = eVec
		invCovariance[tag] = invCov
		
		# Calculates mahal distance for each measure.
		for meas in classMeasures:
			mDist = mahal.calcMahalDistance(meas, invCov, eVec)
			mahalDistances[tag].append(mDist)
			mahalDistances[tag].sort()

	{tag: mahalDistances[tag].sort() for tag in mahalDistances}  # noqa: E501
	
	return Classifier(featureTable, mahalDistances, invCovariance, eigenVectors)  # noqa: E501
