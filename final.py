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
"""
import argparse
import reader
import feature
import classifier
import evaluation


def main(train, test, theta, f, p0):
	trainMeas, trainTags = reader.readData(train)
	testMeas, testTags = reader.readData(test)

	# Table for looking up features for classifying.
	idealFeatures = feature.selectFeatures(trainMeas, trainTags, testMeas, testTags, theta, p0)  # noqa: E501

	tagger = classifier.buildClassifier(trainMeas, trainTags, idealFeatures, f)
	print(tagger.features)
	
	results = []
	for meas in testMeas:
		results.append(tagger.tag(meas, p0))
	print(evaluation.calcAccuracy(results, testTags))
	# conMatrix = evaluation.calcConMatrix(results, testTags)

	# ccAccuracy = evaluation.calcClassCondAccuracy(conMatrix)
	# print(ccAccuracy)

	# accuracy = evaluation.calcAccuracy(conMatrix)
	# print(accuracy)

	# classCondReserve = evaluation.calcClassCondReserve(conMatrix)
	# print(classCondReserve)

	# rejectRate = evaluation.calcReject(conMatrix)
	# print(rejectRate)
	return


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Loads data from file and hyperparameters for testing.")  # noqa: E501
	
	parser.add_argument('train', type=str, help="File containing data for training classifier.")  # noqa: E501
	parser.add_argument('test', type=str, help="File containing data for testing classifier.")  # noqa: E501
	parser.add_argument('featureThreshold', type=float, help="""Threshold value for determining if
		measurement feature is necessary. Used as fixed default if --optimize is also
		specified.""")
	parser.add_argument('subspaceThreshold', type=float, help="""Theshold
		eigenvalue for inclusion in subspace. Measurements default to reserve class
		if below. Used as fixed default if --optimize is also specified.""")
	parser.add_argument('classThreshold', type=float, help="""Theshold
		value for permitting classification. Measurements default to reserve class if
		below. Used as fixed default if --optimize is also specified.""")
	parser.add_argument("-o", "--optimize", action='store_true',
	 help="""Specifies if program should exploratory search for optimal
	hyperparameters. If flagged, all parameters are used for defaults in search.""")  # noqa: E501
	args = parser.parse_args()

	main(args.train, args.test, args.featureThreshold, args.subspaceThreshold, args.classThreshold)  # noqa: E501
