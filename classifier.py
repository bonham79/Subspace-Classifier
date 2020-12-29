"""
Module for building subspace classifier.

"Classifier" object calculates subspaces for each class and projects measurements into
subpsace. Mahalanobis distance from mean of subspace is calculated through 'tag'  # noqa: E501
to provide a classification label.

'buildClassifier' processes the measurements and parameters necessary for creation
of 'Classifier' object.
"""
import mahal
import feature
import numpy as np
from typing import Dict, List, Any, Tuple


class Classifier:
	"""
	Classifier for tagging values. Projects centered measurements into subspaces
	trained for each class. Space that projected measures are closest to mean are
	considered candidate classes.

	Inputs:

		mahalDistances: Class indexed dictionary of ordered mahalanobis distances.
		Serve as distribution of instances in subspace.

		means: Class indexed dictionary of mean vectors for each class. Used to adjust  # noqa: E501
		measures to be centered around origin of projected space.

		invCovariance: Class indexed dictionary of inverse Covariance matrices for
		calculation of mahalanobis distances. Matrices of size 0 are assumed to come
		from distinguishing features. i.e. They are measurements that automatically cateogirze
		space.

		eigenVectors: Class indexed dictionary of vectors for projecting measures to subspaces of
		respective indexes.

		featureTable: Class indexed dictionary of lists of ints that specify which feature indexes
		of measurements should be used for training for respective classes.
	"""

	def __init__(self, mahalDistances: Dict[str, List[float]], means: Dict[str, List[float]],  # noqa: E501
		invCovariance: Dict[str, Any], eigenVectors: Dict[str, Any],
		featureTable: Dict[str, List[int]]) -> List[float]:

		self.distances = mahalDistances
		self.vectors = eigenVectors
		self.invCovariance = invCovariance
		self.features = featureTable
		self.means = means

	def tag(self, meas: Tuple[float], classThreshold: float):
		""" Classifies measurement according to mahalanobis distance metric after
			projecting measurement to subspaces of each class through PCA. Measurements
			are reduced to only specified essential features in 'features.' Then reduced
			measurements are projected to each class' subpsace through 'vectors.' Distance  # noqa: E501
			is then calculated from each projection to the mean of projected subspace.
			Subspace with smallest probability of random occurence in subspace are
			considered as candidates. Failure to meet parameter restrictions leads
			to classification in REJECT class.

		Inputs:

			meas: Tuple of measurements for classification.

			classThreshold: Upper-bound of null hypothesis to be accepted.
			Failure to remain under leads to rejection of class as possible candidate.

		Outpus:

			tag: String specifying likely class. Failure for any class to beat
			threshold results in a REJECT tag.
		"""

		# Initializing defaults.
		maximum = classThreshold
		tag = "RESERVE"

		for classValue in self.features:
			vect = self.vectors[classValue]
			invCov = self.invCovariance[classValue]

			# Narrows down measurements to essential features. Assumes ordered.
			reducedMeasures = [meas[feat] for feat in self.features[classValue]]
	
			# Converts for numpy.
			measArray = np.asmatrix(reducedMeasures)

			# If the invCovariance is 0, then covariance is 0. As such, we only know
			# this class by a specific measurement set.
			if invCov is None:

				# Since we charactorize the defining vector as the measure, we just need
				# to see if they are the same.
				if meas is vect:
					return classValue

				# If it is not the same, we cannot label this class and must proceed
				# with next.
				else:
					continue

			# Adjust by means
			measArray = measArray - self.means[classValue]

			# Calculate mahalanobis distance
			dist = mahal.calcMahalDistance(measArray, invCov, vect)

			# Find likliehood of null-hypothesis for this class.
			py = mahal.calcMahalProb(dist, self.distances[classValue])

			# Checks if it is minimum.
			if py < maximum:
				# Sets as tag and values to beat.
				tag = classValue
				maximum = py
				
		return tag


def buildClassifier(reducedMeasures: Dict['str', List[float]],
	featureTable: Dict['str', List[int]], eigenThreshold: float) -> Classifier:
	"""
Function for processing data and building mahalanobis
classifier. Identifies principle components for
subspace of each class and then constructs classifier
that can project new values to estimate class.

Inputs:

	reducedMeasures: dict of class measurement pairs. Measurements
	are reduced to solely essential features for respective class.

	featureTable: dict of classes as keys and tuples of essential feature indexes
	as values. (Just here to pass to the classifier proper.)

	eigenThreshold: float indicating fraction of variance to be
	spanned by eigen vectors during decomposition.

	Outputs:

	 	tagger: A classifier that projects measures
	 	to subspace defined by each class and uses mahalanobis distances to
	 	calculate likelihood of inclusion in class.
	"""

	# Groups measures by classes.
	mahalDistances = {val: [] for val in reducedMeasures}
	eigenVectors = {val: None for val in reducedMeasures}
	invCovariance = {val: None for val in reducedMeasures}
	classMeans = {val: None for val in reducedMeasures}

	# For each class.
	for tag in reducedMeasures:

		# Chooses measurements
		classMeasures = reducedMeasures[tag]
		
		# Converts to array for numpy use
		classMatrix = np.asmatrix(classMeasures)

		# Calculate means and stores
		means = np.mean(classMatrix, axis=0)
		classMeans[tag] = means

		# Adjusts
		classMatrix = classMatrix - means

		# Identifies essential eigen vectors and values for class.
		invCov, eVec = mahal.eigenDecomp(classMatrix, eigenThreshold)
		eigenVectors[tag] = eVec
		invCovariance[tag] = invCov
		
		# Calculates mahal distance for each measure.
		for meas in classMeasures:
			mDist = mahal.calcMahalDistance(np.asmatrix(meas), invCov, eVec)
			mahalDistances[tag].append(mDist)

		# Sorts tags.
		mahalDistances[tag].sort()
	
	return Classifier(mahalDistances, classMeans, invCovariance, eigenVectors, featureTable)  # noqa: E501

def main():

	# Tests building the classifier.
	measures = [[1, 2, 3], [3, 2, 1], [1, 2, 3], [2, 2, 2], [3, 2, 1], [2, 3, 1]]  # noqa: E501
	tags = ["dog", "dog", "cat", "dog", "cat", "cat"]
	features = {"dog": [0, 1],  "cat": [0, 2]}
	reduc = feature.reduceFeatures(measures, tags, features)

	matrix = np.asmatrix([[1, 2], [3, 2], [2, 2]])  # This is dog.
	covar = np.cov(matrix, rowvar=False) # Just allowing us to see what the dog should be.
	print(covar)
	print(np.linalg.eig(covar))


	tagger = buildClassifier(reduc, features, 1.0)
	print(tagger.distances)  # Dog should be around 1, 4, 10
	print(tagger.vectors)
	print(tagger.invCovariance)
	print(tagger.tag([2, 2, 0], .9))

if __name__ == '__main__':
	main()