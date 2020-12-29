"""
Module for identifying essential features in data-set.

'selectFeatures': creates a look up table of essential
features for each class index by constructing classifiers
for each iteration of class and feature, selecting features
that have sufficient accuracy determined by a threshold.

'reducefeatures' takes a essential feature look-up table
and pairs of measures and class tags to create a look-up
table of only essential measures for each class.

'calcClassAccuracy' calculates the class conditional accuracy
for each class to determine which class is best performance.

'ZClassifier' is a module specific iteration of the Mahalanobis
classifier tailored exclusively for features (as one dimensional
classification can be optimized as simply a Z-score calculation).
"""
import mahal
import numpy as np
from typing import List, Tuple, Dict


def calcClassAccuracy(assigned: List[str], real: List[str]) -> Dict[str, float]:  # noqa: E501
	"""
	Function for calculating class conditional probabilities for given feature.
	Assumes all possible values are found in real. Then calculates the probability  # noqa: E501
	of correct identification for each class.

	Inputs:

		assigned: list of class values determined by classifier.

		real: list of real class values corresponding to above's indexes.

	Outputs:

		ClassAccuracy: Dictionary of class keys and floats identify accuracy in
		identification for respective classes.
	"""
	listLength = len(assigned)
	possibleValues = set(real)  # Concerned with real values.

	counts = {value: 0 for value in possibleValues}
	totals = {value: 0 for value in possibleValues}

	# Searches along lists
	for i in range(listLength):
		tag = real[i]

		# Keeps running count of occurence of tag.
		totals[tag] += 1

		# And counts the amount of times assignment was correct.
		if assigned[i] == tag:
			counts[tag] += 1

	# Returns dictionary with probability of correct assignment for each class.
	return {value: counts[value]/totals[value] for value in possibleValues}


def selectFeatures(trainMeas: Tuple[Tuple[float]], trainTags: Tuple[str],
	testMeas: Tuple[Tuple[float]], testTags: Tuple[str],
	featureThreshold: float) -> Dict[str, List[int]]:  # noqa: E501
	"""
Function for identifying essential features for class tagging.
produces a class look-up table of essential features for given class.

Inputs:

	trainMeas: tuple of tuples for training classifier for feature detection.

	trainTags: tuple of strings for tags associated with above. Assumes index
	i refers to same measure and tag pair. (e.g. trainTags[i] is class for
	trainMeas[i]).

	testMeas: tuple of tuples for testing classifier accuracy with individual
	features.

	testTags: tuple of strings for tags associated with above. Same index
	assumption as above.

	featureThreshold: minimum accuracy for feature to be considered essential
	for class.

	classThreshold: Maximum proportion of Mahalanobis distances for classifier to
	tolerate before choosing RESERVE class.

Outputs:

	featureTable: Dict with classes as keys with values of lists of integers
	specifying which feature indexes are essential for classification.

	"""

	# Assumes that all features are uniform length.
	numFeatures = len(trainMeas[0])

	# Initializes look-up table to store feature outcomes.
	tagValues = set(trainTags)
	featureTable = {tag: {feature: 0.0 for feature in range(numFeatures)} for tag in tagValues}  # noqa: E501

	# For each feature
	for feature in range(numFeatures):

		# Creates feature look-up table for single feature.
		singleFeature = {tag: [feature] for tag in tagValues}

		# Reduces measurements to only this feature.
		reducMeasures = reduceFeatures(trainMeas, trainTags, singleFeature)

		# One feature has 1x1 covariance, so projection is arbitrary.
		# As such, we just calculate a 1x1 mahal, which is squared z-score.
		dists = {tag: [] for tag in tagValues}
		means = {tag: 0.0 for tag in tagValues}

		for tag in reducMeasures:
			meas = reducMeasures[tag]

			# Unpacking lists.
			meas = [m[0] for m in meas]

			# Calculate means and variance.
			means[tag] = sum(meas) / len(meas)
			var = np.var(meas)

			# If we have no variance, we'll just classify distance as a single 0.
			if var == 0:
				dists[tag].append(None)

			# Calculate mahal distances (squared z-score).
			else:
				# Otherwise normal.
				for n in meas:
					dists[tag].append((n-means[tag])**2 / var)

				# Sorting
				dists[tag].sort()

		# Creates classifier off these measures.
		tagger = ZClassifier(dists, means)

		# Tests classifier on each measure in the testing values and adds to results.
		results = []

		for test in testMeas:
			results.append(tagger.tag(test[feature]))

		# Calculates accuracy for each class.
		probs = calcClassAccuracy(results, testTags)

		# Tracks probabilities in table.
		for tag in probs:
			featureTable[tag][feature] = probs[tag]

	# Table for tracking best features.
	bestFeatures = {}

	for tag in featureTable:

		# Tracks best feature and features that surpass threshold.
		best = 0
		winners = []

		# Working iteratively through features. Ordered outcome.
		for feat in range(numFeatures):

			# Sees if meets threshold value.
			if featureTable[tag][feat] > featureThreshold:
				winners.append(feat)

			# Sees if best performer.
			if featureTable[tag][feat] > featureTable[tag][best]:
				best = feat

		# If we have a non-empty list, we can take that.
		if winners:
			bestFeatures[tag] = winners

		# Empty list, we take what we can.
		else:
			bestFeatures[tag] = [best]


	return bestFeatures


def reduceFeatures(measures: Tuple[Tuple[float]], tags: Tuple[str], features: Dict[str, List[int]]) -> Dict[str, List[List[float]]]:  # noqa: E501
	"""
Function that takes 'features' look-up table and returns a look-up table
of reduced measurements for each class.

input:

	measures: Tuple of measurement tuples for reduciton.

	tags: tuple of associated string values for above measures.

	features: look up table with tags as keys and indexes of essential features
	for these classes as values.

output:

	reducedMeasures: dictionary with tags as keys and associated measurements
	reduced to the ideal features for those tags.
	"""

	# Initializes dictionary for look-up table.
	spaces = {val: [] for val in features}
	# First we reduce measurements according to their essential features
	# for each class and place them into a table organized by classes.
	for i in range(len(measures)):

		# Locates class-measure pair
		val = tags[i]
		meas = measures[i]

		# Looks up essential features for class.
		essFeatures = features[val]

		# Reduces measurement to only essential features
		reducMeasure = [meas[index] for index in essFeatures]

		# Places measurement in look-up table.
		spaces[val].append(reducMeasure)

	return spaces

class ZClassifier:
	"""A classifier for feature accuracy. Used instead of classifier.py due to
	troubleshooting needs
	""" 
	def __init__(self, distances, means):
		self.distances = distances
		self.means = means

	def tag(self, measure):

		# Setting variables.
		bestTag = None
		bestVal = 1

		# iterating over each class.
		for tag in self.distances:
			dist = self.distances[tag]
			m = self.means[tag]
			adjMeas = measure - m

			# If we have 0 dist, there was no variance for this class
			# so we assume this feature was distinguishing.
			if dist is None:

				# We check if the measure is the same as the mean (therefore same as rest)
				if adjMeas == 0:
					return tag

				# If not we move on. (No way to figure out space for this class.)
				else:
					continue

			py = mahal.calcMahalProb(adjMeas, dist)

			if py < bestVal:
				bestTag = tag
				bestVal = py

		return bestTag

def main():
	# Test calcClassAccuracy
	results = [3,2,0,5,5]
	real =    [3,2,3,0,5]

	# Should print out:
	ideal = {3: .5, 2: 1.0, 5: 1.0, 0: 0}

	assert calcClassAccuracy(results, real) == ideal

	# Test reduceFeatures
	measures = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
	tags = ["dog", "cat", "dog"]
	features = {"dog": [1, 0], "cat": [0]}
	print(reduceFeatures(measures, tags, features))  # Should line up with feature table.  # noqa: E501

	# Tests selectFeatures.

	# Let's create a trivial problem where antime the third
	# value exceeds 5 it's a dog, but cat when less.
	# We'll make the other values random.
	import random

	# random.seed(42)

	# We'll randomize all these values.
	trainMeas = [(random.uniform(0, 1000), random.randrange(0, 12), random.uniform(0, 1000))  # noqa: E501
	for _ in range(10000)]

	# Then assign values based only off the third.
	trainTags = ["dog" if meas[1] <= 5 else "cat" for meas in trainMeas]

	# And for testing...
	testMeas = [(random.uniform(0, 1000), random.randrange(0, 12), random.uniform(0, 1000))  # noqa: E501
	for _ in range(10000)]
	testTags = ["dog" if meas[1] <= 5 else "cat" for meas in testMeas]

	print(selectFeatures(trainMeas, trainTags, testMeas, testTags, .9))


if __name__ == '__main__':
	main()
