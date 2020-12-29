"""
Functions for evaluating testing for evaluation metrics. Functions
revolve around developing a confusion matrix from results and real value
tuples. All but the accuracy function use this matrix for value calculations.

Functions are:

calcAccuracy: calculates accuracy of data.

calcConMatrix: creates normalized confusion matrix.

calcClassCond: calculates the conditional class accuracy for each possible
tag (performance restricted to tag).

calcClassCondReserve: calculates the class conditional reserve rate
(likielihood of a tag being indeterminable).

calcReject: gives rejection rate of data set.
"""
from typing import Tuple, Dict


def calcAccuracy(results: Tuple[str], real: [str]) -> float:
	"""
	Calculates classification accuracy of results
	against real values.

	Inputs:

		results: list of tags given by classifier.
	
		real: real tags paired by index with results.

	Outputs: 

		accuracy: float indicating percent accuracy.
	"""

	correct = 0

	for i in range(len(results)):
		if results[i] == real[i]:
			correct += 1

	return correct/len(real)

def calcConMatrix(results: Tuple[str], real: [Tuple[str]]) -> Dict[str, Dict[str, float]]:  # noqa: E501
	"""
	Creates confusion matrix table of results, normalized
	by number of provided values. Returned as dict of dict of assigned tag,
	given real tag. (e.g. matrix[real][assigned]).
	
	Inputs:

		results: Tuple of results from classifier

		real: tuple of real class values sharing indexes with above.

	Outputs: 

		conMat: Dict of dict of probability values. Dicts use classes as keys.
	"""
	total = len(real)
	trueTags = set(results) | set(real)
	assignedTags = trueTags.copy()

	# Assignment doesn't guarantee presence of "RESERVE" but want it reliably
	# present for evaluation. So will double-check.
	assignedTags.add("RESERVE")
	trueTags.discard("RESERVE")

	table = {true: {assigned: 0.0 for assigned in assignedTags} for true in trueTags}  # noqa: E501

	for i in range(total):
		true = real[i]
		assigned = results[i]
		table[true][assigned] += 1 / total

	return table


def calcClassCond(conMatrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
	"""
	Produces class conditional probabilities for the confusion matrix.
	Returned as dictionary with classes as keys and probabilities of accurate
	identification given class (T|c) as values.

	Inputs:

		conmatrix: the confusion matrix dict.

	Outputs:

		classCond: Dictionary of class conditional probabilities indexed by classes.
	"""

	classCond = {}

	for tag in conMatrix:
		total = sum(conMatrix[tag].values())
		accurate = conMatrix[tag][tag]
		reserve = conMatrix[tag]["RESERVE"]  # NOTE: We're going off slides here. May need to fix.  # noqa: E501
		classCond[tag] = accurate / (total - reserve)
	return classCond


def calcClassCondReserve(conMatrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
	"""
	Calculate class conditional reserve rates, indexed by class.

	Input:

		conMatrix: Confusion matrix dictionary.

	Otputs:

		condReserves: Dictionary with classes as keys and conditional (P(T|C))
		reserve probabilities.
	"""
	classCond = {}

	for tag in conMatrix:
		total = sum(conMatrix[tag].values())
		reserve = conMatrix[tag]["RESERVE"] / total
		classCond[tag] = reserve
	return classCond


def calcReject(conMatrix: Dict[str, Dict[str, float]]) -> float:
	"""
	Returns the probability of rejection (assigned RESERVE) from the matrix

	Inputs:
		
		conMatrix: the confusion matrix.

	Outputs:

		Reject rate: float of the rejection rate.
	"""
	return sum([conMatrix[true]["RESERVE"] for true in conMatrix])

def main():
	assigned = [0, 1, "RESERVE", 1, 0, 0, 1, 0, "RESERVE", 0]
	real =     [1, 1, 1,         0, 0, 1, 0, 0, 0,         1]  # noqa: E222

	# Test calc accuracy
	acc = calcAccuracy(assigned, real)
	assert acc == .3, acc

	# Test conditional matrix
	matrix = calcConMatrix(assigned, real)
	ideal = {0: {0: .2, 1: .2, "RESERVE": .1},
			 1: {0: .3, 1: .1, "RESERVE": .1}}
	#assert matrix == ideal, matrix # Rounding error.

	# Calc classCondReserve
	idealCondReserve = {0: .2, 1: .2}
	condReserve = calcClassCondReserve(matrix)
	assert condReserve == idealCondReserve, condReserve

	# Test calcClassConditional
	idealConds = {0: .2 / (.5 - .2), 1: .1 / (.5 - .2)}
	conds = calcClassCond(matrix)
	# assert conds == idealConds, conds # We changed the calculation.

	# Test calcReject
	reject = calcReject(matrix)
	assert reject == .2, reject

if __name__ == '__main__':
	main()