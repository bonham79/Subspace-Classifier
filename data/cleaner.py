import csv

with open("data.csv") as file:
	read = csv.reader(file)
	for row in read:
		cleaned = [row[0]] + row[2:6] + row[7:12] + [row[13]] + row[15:]
		cleaned[-1] = str(((2020 - int(cleaned[-1]))+9)//10)
		print(",".join(cleaned))