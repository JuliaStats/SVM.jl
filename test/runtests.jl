srand(1)
using SVM
iris = readcsv(joinpath(dirname(@__FILE__), "iris.csv"))
X = iris[:, 1:4]'
p, n = size(X)
train = randbool(n)
Y = [species == "setosa" ? 1.0 : -1.0 for species in  iris[:, 5]]
model = svm(X[:,train],Y[train])
@assert (predict(model, X[:, ~train])) == Y[~train]
