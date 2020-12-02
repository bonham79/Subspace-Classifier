"""
Module for identifying essential features in data-set.
function 'selectFeatures' forms a mahalanobis classifier
based on each individual features and identifies which features
of each class are capable of reaching the threshold value for
acceptance. 'reduceFeatures' is called to convert measurements
to their reduced form and arrange them by class in a dict.
"""
import classifier
from typing import List, Tuple, Dict


def calcClassAccuracy(assigned: List[str], real: List[str])-> Dict[str, float]:
	"""
		Function for calculating class conditional probabilities for given feature.
		Assumes all possible values are found in real. Then calculates the probability  # noqa: E501
		of correct identification for each class.
	"""
	listLength = len(assigned)
	assert listLength == len(real)

	possibleValues = set(real)

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

def selectFeatures(trainMeas: Tuple[Tuple[float]], trainTags: Tuple[str], testMeas, testTags, featureThreshold: float, classThreshold: float) -> Dict[str, List[int]]:  # noqa: E501
	"""
	Function for identifying essential features for class tagging.
	Outputs a dict with classes as keys and list of essential features for values.  # noqa: E501
	measures is set of measurements from data, tags is set of tags for data.
	figureThreshold is the minimum accuracy required for feature to be deemed essential.
	classThreshold is the threshold for identification of a class for the classifier.
	"""

	# Splits data for testing and training.
	numFeatures = len(trainMeas[0])

	# Initializes keys for dictionary of class essential features. Assumes only
	# classes present in data are possible.
	tagValues = set(trainTags)
	essentialFeatures = {tag: [] for tag in tagValues}

	# For each feature
	for feature in range(numFeatures):
		# Creates feature dictionary for single feature.
		singleFeature = {tag: [feature] for tag in tagValues}

		# Creates classifier.
		tagger = classifier.buildClassifier(trainMeas, trainTags, singleFeature, 0)  # Since there's only one eigenvector.  # noqa: E501

		# Tests classifier
		results = []
		for meas in testMeas:
			results.append(tagger.tag(meas, classThreshold))

		# Calculates accuracy for each class.
		probs = calcClassAccuracy(results, testTags)

		# Assigns feature to class assignments exceeding threshold.
		for i in probs:
			if probs[i] > featureThreshold:
				essentialFeatures[i].append(feature)

	# If no features were chosen we'll need to use all.
	for tag in essentialFeatures:
		if not essentialFeatures[tag]:
			essentialFeatures[tag] = [feature for feature in range(numFeatures)]
	return essentialFeatures

def reduceFeatures(measures: Tuple[Tuple[float]], tags: Tuple[str], features: Dict[str, List[int]]) -> Dict[str, List[List[float]]]:  # noqa: E501
	"""
	Function that takes 'features' look-up table and returns a look-up table
	of reduced measurements for each class.
	"""
	spaces = {val: [] for val in features}
	size = len(measures)

	# First we reduce measurements according to their essential features
	# for each class and place them into a table organized by classes.
	for i in range(size):
		# Locates class-measure pair
		value = tags[i]
		measure = measures[i]

		# Looks up essential features for class.
		essFeatures = features[value]

		# Reduces measurement to only essential features
		reducMeasure = [measure[index] for index in essFeatures]

		# Places measurement in look-up table.
		spaces[value].append(reducMeasure)
	return spaces

def main():
	meas = [(2, 0, 3), (1, 0, 5), (0, 1, 5), (0, 3, 6)]
	tags = ["dog", "dog", "cat", "cat"]

	i = 0
	while i < 1:
		print(selectFeatures(meas, tags, i, 0.99))
		i += .1


if __name__ == '__main__':
	main()