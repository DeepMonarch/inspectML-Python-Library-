from inspectML.evaluation import classification, regression

# For regression
print(regression.mse([1,2,3,4], [2,3,4,5]))
print(regression.mae([1,2,3,4], [2,3,4,5]))
print(regression.rmse([1,2,3,4], [2,3,4,5]))
print(regression.r2([1,2,3,4], [2,3,4,5]))

# For classification
X = [[0, 1], [1, 0], [2, 1], [3, 2]]
labels = [0, 0, 1, 1]
print(classification.SIL(X, labels))
print(classification.ARI([0, 0, 1, 1], [0, 1, 1, 1]))