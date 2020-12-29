import csv

include = ["Classical", "Rock", "Jazz", "Country"]
with open("data.csv") as file:
	read = csv.reader(file, delimiter=",")
	for row in read:
		if row[0] in include:
			cleaned = row[4:10] + row[11:13] + row[14:16] + [row[-1]] + [row[0]]
			print(",".join(cleaned))
		# print(row[-1])
		# if row[-1] in include:
		# 	print(row)
