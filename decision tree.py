from sklearn import tree
import csv
import random
from sklearn.externals.six import StringIO
import pydot
from IPython.display import Image
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

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
X, Y= splitDataset(dataset)

A, B= splitDataset(dataset)


#test accuracy
def test_depth(max_depth, X_train, X_test, Y_train, Y_test):
    depths = []
    accuracy=[]
    for depth in range(2, max_depth + 1):
        depths.append(depth)
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth)
        clf.fit(X_train, Y_train)
        test_result=clf.predict(X_test)
        accuracy.append(accuracy_score(test_result, Y_test))


    pyplot.plot([number for number in range(2, max_depth+1)], accuracy, '-r', label="accuracy")
    pyplot.legend()
    pyplot.xlabel('depth')
    pyplot.ylabel('accuracy')
    pyplot.title('accuracy with the increase of depth')
    pyplot.show()

test_depth(100, X, A, Y, B)


#shu zhuang tu
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
clf.fit(X, Y)
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,filled=True, rounded=True, special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("quality of wine.pdf")
