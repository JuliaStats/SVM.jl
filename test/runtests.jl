using SVM
using Base.Test
srand(1)
iris = readcsv(joinpath(dirname(@__FILE__), "iris.csv"))
X = map(Float64, iris[:, 1:4]')
p, n = size(X)
train = rand(Bool, n)
Y = [species == "setosa" ? 1.0 : -1.0 for species in iris[:, 5]]
model = svm(X[:,train], Y[train])
@test predict(model, X[:, ~train]) == Y[~train]
