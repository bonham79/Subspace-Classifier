"""
Module for reading and preparing data files. Assumes all files are csvs.

Functions are:

read: helper function accepting string specifying file to convert into a list
of list values.

readData: main function for reading in files. Shuffles and typecasts data before  # noqa: E501
outputing a tuple of float tuples for measurements and a tuple of strings as class tags.
"""
import csv
import random
from typing import Tuple, List


def read(file: str) -> List[str]:
	"""Helper function for reading in files.
		
	Inputs:
		file: string specifying file path.

	Output:

		Measures: List of strings.
	"""
	items = []
	with open(file) as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			items.append(row)
	return items


def readData(file: str) -> Tuple[Tuple[float], str, Tuple[float], str]:
	"""Function for reading in a csv file to separate into measurements and tags.
	   Prepares data by typecasting measurements as tuples of floats and tags
	   as strings, splitting in half for testing and training data.
	   Shuffles data to ensure randomness for testing. Assumes all data is preprocessed
	   so class tag is final item and all else are measures.
	
	Inputs:

		file: string designating file path

		seed: integer for replicaple random seeding.
	
	Output:

		TrainMeas: List of tuples for training data.

		TrainTags: List of class strings for training.

		TestMeas: List of tuples for testing data.
		
		TestTags: List of class strings for testing.
	"""

	meas = []
	tags = []

	# Reads in data and shuffles to avoid dependencies.
	data = read(file)
	random.shuffle(data)

	for pair in data:
		# If data has formatting problems or missing values we'll do error handling.
		try:
			# Makes sure measurements are all type-cast as floats.
			vals = []
			for item in pair[:-1]: # excludes last term as that's the class tag.
				vals.append(float(item))

			# Separates measure from pairs.
			tags.append(str(pair[-1]))

			# Converts to tuple for stability. Comes last after casting has been
			# attempted to avoid errors.
			meas.append(tuple(vals))
			
		except ValueError:
			continue

	# Splitting
	trainMeas = meas[: len(meas)//2]
	testMeas = meas[len(meas)//2:]
	trainTags = tags[: len(tags)//2]
	testTags = tags[len(tags)//2:]

	# Makes sure nothing went wrong.
	assert len(trainMeas) == len(trainTags) and len(testMeas) == len(testTags)
	
	return trainMeas, trainTags, testMeas, testTags


if __name__ == '__main__':

	# Checking order is preserved. (Also lengths if there's no indexing error.)
	meas, tags, meas2, tags2 = readData('test.txt', 8)

	measures = meas + meas2
	tagging = tags + tags2

	pairs = [(measures[i], tagging[i]) for i in range(len(measures))]
	pairs = set(pairs)

	meas, tags, meas2, tags2 = readData('test.txt', 20)

	measures = meas + meas2
	tagging = tags + tags2

	pairs2 = [(measures[i], tagging[i]) for i in range(len(measures))]
	pairs2 = set(pairs2)

	assert pairs | pairs2 == pairs, pairs | pairs2  # noqa: W292