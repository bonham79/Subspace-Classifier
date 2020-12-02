"""
Module for reading and preparing data files. Assumes all file names specify csv files.

Functions are as follows:

read: helper function accepting string specifying file to convert into a list
of list values.

readData: main function for reading in files. Shuffles and typecasts data before  # noqa: E501
outputing a tuple of float tuples for measurements and a tuple of strings as class tags.
"""
import csv
import random
from typing import Tuple, List


def read(file: str) -> List:
	# Helper function for reading in files.
	items = []
	with open(file) as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			items.append(row)
	return items

def readData(file: str) -> Tuple[Tuple[float], str]:
	"""Function for reading in a csv file to separate into measurements and tags.
	   Prepares data by typecasting measurements as tuples of floats and tags
	   as strings. Shuffles data to ensure randomness for testing.
	"""  # noqa: E501
	meas = []
	tags = []

	# Reads in data and shuffles to avoid dependencies.
	data = read(file)
	random.shuffle(data) 

	for pair in data:
	# Separates measure from pairs.
		tags.append(str(pair[-1]))

	# Makes sure measurements are all type cast as floats.
		vals = []
		for item in pair[:-1]:
			vals.append(float(item))

	# Converts to tuple for stability.
		meas.append(tuple(vals))

	# Returns as tuple for stability.
	return tuple(meas), tuple(tags)


if __name__ == '__main__':
	meas, tags = readData('test.txt')
	assert meas == ((1.0, 0.0, 3.0), (2.0, 3.0, 2.0), (2.0, 3.1, 0.3)), meas
	assert tags == ("dog", "cat", "dog"), tags
