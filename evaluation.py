"""
functions for evaluating testing for evaluation metrics.
"""
from typing import Tuple

def calcAccuracy(results, real):
	correct = 0
	for i in range(len(results)):
		if results[i] == real[i]:
			correct += 1
	return correct/len(real)