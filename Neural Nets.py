import csv
import random
import neurolab as nl
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
    b=zip(b)
    return [a, b]
X, Y= splitDataset(dataset)
A, B= splitDataset(dataset)

#node number denote the number of nodes in hiden layer
accuracy=[]
for node_number in range(100,150):
    # i neurons for hidden layer, 1 neuron for output
    # 2 layers including hidden layer and output layer

    net=nl.net.newff([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]],[node_number,1])
    net.trainf = nl.train.train_rprop
    net.train(X, Y, epochs = 100)
    test_result= net.sim(A)
    print test_result
    accuracy.append(accuracy_score(test_result, B))
print accuracy
