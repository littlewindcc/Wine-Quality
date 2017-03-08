import csv
import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#load file
lines=csv.reader(open('final.csv',"rb"))
dataset=list(lines)

for i in range(len(dataset)):
    dataset[i]=[float(x) for x in dataset[i]]

#seperate dataset
def splitDataset(dataset):
    splitRatio = 0.67
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
	    index = random.randrange(len(copy))
	    trainSet.append(copy.pop(index))
    a=[row[0:11] for row in trainSet]
    b=[row[-1] for row in trainSet]
    return [a, b]

X, Y=splitDataset(dataset) #training data
A, B=splitDataset(dataset) #testing data


M=[0.01,0.1,1,10,100,1000,10^4,10^5,10^6,10^7]
for C in M:
    clf = SVC(C)
    clf.fit(X, Y)
    test_result=clf.predict(A)
    print accuracy_score(test_result, B)
