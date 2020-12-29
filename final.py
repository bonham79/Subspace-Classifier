# To-do: eigh is symmetric and ordered so remove sorting. check first
# see what happens to accuracy when mean is subtracted from y instead of x. (Real distance in subspace.) Also check to see if y mean is 0
# review subspaces to see effects of subspaces. Does the mean become the relative 0 point?
# Try various probability point calculations from notes.
# Make f stop at a certain percentage significance. (.005?)
# Add a way to just check one value.
"""
Command line program for training a mahalanobis subspace classifier and testing
on a data set. Outputs evaluation data.

Command line inputs:

- train: file name of data containing desired training data. Assumes file is
csv with every line being a sequence measurement features and class label iss
final value. (e.g. '1,3,2,dog' is measurement values '1,3,2' of class 'dog'.)

- test: file name of data containing desired data for testing. Is assumed
csv in format such as above.

- featureThreshold: parameter used for determining which features are critical
in testing. Is of value (0,1). Lower values increases inclusivity of data at
cost of performance. Higher values decrease inclusivity for higher performance.

- subspaceThreshold: parameter used for determining number of bases in subspace.  # noqa: E501
Is of value (0,1). Ratio is inversely proportional to inclusivity of subspace.

- classThreshold: parameter used to determine whether measurement is not placed into
reserve class. Is of value (.5,]. Is directly proportional of inclusivity of measurements.

- seed: Value for seeding randomized shuffle of data. For recreating tests.
"""
import argparse
import reader
import feature
import classifier
import evaluation
import random
from numpy import arange
from typing import List, Tuple
from matplotlib import pyplot as plt
from classifier import Classifier

def build(data: str, theta: float, f: float):
	"""
	Trains a Mahalanobis classifier on given data set. Data is split and
	shuffled by seed value into testing and training sets. Ideal features are identified by  # noqa: E501
	theta parameter and then reduced to a subspace defined by f fraction.
	Distances within the mean of this subspace are identified and then used
	to produce a classifier.

	data: string identifying file path to data.
	theta: float used for feature identification.
	f: float used for principal components.
	seed: int used for random seed.

	Output: Classifier class object, testing measures, and testing tags.
	"""

	# Reads in data and outputs randomized values.
	trainMeas, trainTags, testMeas, testTags = reader.readData(data)

	# Creates table identifying ideal features for each class.
	idealFeatures = feature.selectFeatures(trainMeas, trainTags, testMeas, testTags, theta)  # noqa: E501

	# Reduces measures for training only on essential measures.
	reducedMeasures = feature.reduceFeatures(trainMeas, trainTags, idealFeatures)  # Change  # noqa: E501

	# Train classifier off these measures.
	tagger = classifier.buildClassifier(reducedMeasures, idealFeatures, f)

	return tagger, testMeas, testTags

def classify(tagger: Classifier,
	measures: Tuple[Tuple[float]], p0: float) -> List[str]:
	"""
	Runs commands for generating tags for given measures. Returns
	list of results.

	tagger: Classifier object
	measures: list of measurement tuples
	p0: float values to determine threshold for rejecting values

	output: list class strings.
	"""

	# Initializes results matrix.
	results = []

	# Classifies each measure in testing values.
	for meas in measures:
		results.append(tagger.tag(meas, p0))

	return results

def main(data, theta, f, p0, seed):

	if args.search:
		defaultTheta = .01
		defaultF = .85
		defaultP0 = .75

		bestFeatureThreshold = None
		bestEigenThreshold = None
		bestClassThreshold = None

		pCounts = []

		# Search for ideal feature threshold.################################
		print("Testing features.")
		seed *= 5
		bestAccuracy = 0.0
		matrices = []

		for t in arange(.005, 1.01, theta):
			# We want to keep using one randomization so we actually know the
			# parameter is better (as opposed to just luck of randomization).
			random.seed(seed)
			tagger, measures, tags = build(data, t, defaultF)

			results = classify(tagger, measures, defaultP0)
			acc = evaluation.calcAccuracy(results, tags)
			print("Accuracy: {}. Theta= {}".format(acc, t))

			if acc > bestAccuracy:
				bestFeatureThreshold = t
				bestAccuracy = acc

			if args.graph:
				conmat = evaluation.calcConMatrix(results, tags)
				matrices.append(conmat)

		print("Ideal theta is: {}".format(bestFeatureThreshold))

		if args.graph:
			for tag in conmat:
				print("Charting accuracies and reserves for class {}".format(tag))

				print("Plotting accuracies")
				accuracies = [matrix[tag][tag] for matrix in matrices]
				print(accuracies)
				# plt.plot(accuracies)
				# plt.show()

				print("Plotting reserves")
				reserves = [matrix[tag]["RESERVE"] for matrix in matrices]
				print(reserves)
				# plt.plot(reserves)
				# plt.show()

		# Search for eigenvalue threshold.
		print("Testing eigens.")  ################################
		seed *= 2
		bestAccuracy = 0.0
		matrices = []

		for e in arange(.5, 1, f):
			random.seed(seed)
			tagger, measures, tags = build(data, defaultTheta, e)

			results = classify(tagger, measures, defaultP0)
			acc = evaluation.calcAccuracy(results, tags)
			print("Accuracy: {}. f= {}".format(acc, e))

			if acc > bestAccuracy:
				bestEigenThreshold = e
				bestAccuracy = acc

			if args.graph:
				conmat = evaluation.calcConMatrix(results, tags)
				matrices.append(conmat)

		print("Ideal f-value is: {}".format(bestEigenThreshold))

		if args.graph:
			for tag in conmat:
				print("Charting accuracies and reserves for class {}".format(tag))

				print("Plotting accuracies")
				accuracies = [matrix[tag][tag] for matrix in matrices]
				print(accuracies)
				# plt.plot(accuracies)
				# plt.show()

				print("Plotting reserves")
				reserves = [matrix[tag]["RESERVE"] for matrix in matrices]
				print(reserves)
				# plt.plot(reserves)
				# plt.show()

		# Search for class threshold. ################################
		print("Testing class cutoff.")
		seed *= 2
		bestAccuracy = 0.0
		matrices = []

		# We don't need to make multiple times.
		random.seed(seed)
		tagger, measures, tags = build(data, defaultTheta, defaultF)

		for p in arange(.5, 1.01, p0):
			results = classify(tagger, measures, p)
			acc = evaluation.calcAccuracy(results, tags)
			pCounts.append(acc)
			print("Accuracy: {}. p0= {}".format(acc, p))

			if acc > bestAccuracy:
				bestClassThreshold = p
				bestAccuracy = acc

			if args.graph:
				conmat = evaluation.calcConMatrix(results, tags)
				matrices.append(conmat)

		print("Ideal p0-value is: {}".format(bestClassThreshold))

		if args.graph:
			for tag in conmat:
				print("Charting accuracies and reserves for class {}".format(tag))

				print("Plotting accuracies")
				accuracies = [matrix[tag][tag] for matrix in matrices]
				print(accuracies)
				# plt.plot(accuracies)
				# plt.show()

				print("Plotting reserves")
				reserves = [matrix[tag]["RESERVE"] for matrix in matrices]
				print(reserves)
				# plt.plot(reserves)
				# plt.show()

	else:
		bestFeatureThreshold = theta
		bestEigenThreshold = f
		bestClassThreshold = p0

	random.seed(seed + 1)
	tagger, measures, tags = build(data, bestFeatureThreshold, bestEigenThreshold)

	results = classify(tagger, measures, bestClassThreshold)
	print(tagger.features)

	acc = evaluation.calcAccuracy(results, tags)
	conmat = evaluation.calcConMatrix(results, tags)
	conds = evaluation.calcClassCond(conmat)
	ccr = evaluation.calcClassCondReserve(conmat)
	reject = evaluation.calcReject(conmat)

	# Printing results.
	for tag in conds:
		print("The class conditional accuracy for {} is {}".format(tag, conds[tag]))
		print("The class conditional reserve rate for {} is {}".format(tag, ccr[tag]))
		print("It comprises {} of the test set.".format(
			sum([1 for i in range(len(tags)) if tags[i] == tag]) / len(tags)))

	print("Accuracy is {} with theta = {}, f = {}, and p0 = {}".format(acc,
		bestFeatureThreshold, bestEigenThreshold, bestClassThreshold))

	print("The reject rate is {}".format(reject))

	if args.graph:
		print("Outputting read rate.")
		print(pCounts)
		# plt.plot(pCounts)
		# plt.show()


	print("Confusion Matrix")
	for tag in conmat:
		print(tag, conmat[tag])

	return


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Loads data from file and hyperparameters for testing.")  # noqa: E501
	
	parser.add_argument('data', type=str, help="File containing data for training and testing classifier.")  # noqa: E501
	parser.add_argument('featureThreshold', type=float, help="""Threshold value for increments in
		measurement feature testing.""")
	parser.add_argument('subspaceThreshold', type=float, help="""Theshold
		eigenvalue increment for optimizing in subspace.""")
	parser.add_argument('classThreshold', type=float, help="""Theshold
		value increment for permitting classification.""")
	parser.add_argument('seed', type=int, help="""Seed for random shuffling of data.""")
	parser.add_argument('--search', '-s', action="store_true", help="""Determines whether to interpret
		values as increment functions. If not specified, assumes they are desired values for one classification.""")
	parser.add_argument('--graph', '-g', action="store_true", help="""Determines whether to graph functions.""")
	args = parser.parse_args()

	main(args.data, args.featureThreshold, args.subspaceThreshold, args.classThreshold, args.seed)  # noqa: E501
