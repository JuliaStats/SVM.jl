SVM.jl
======


[![Build Status](https://travis-ci.org/JuliaStats/SVM.jl.svg)](https://travis-ci.org/JuliaStats/SVM.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaStats/SVM.jl/badge.png)](https://coveralls.io/r/JuliaStats/SVM.jl)
[![Package Evaluator](http://iainnz.github.io/packages.julialang.org/badges/SVM_0.3.svg)](http://iainnz.github.io/packages.julialang.org/?pkg=SVM&ver=0.3)


# SVMs in Julia

Native Julia implementations of standard SVM algorithms.
Currently, there are textbook style implementations of
two popular linear SVM algorithms:

* Pegasos (Shalev-Schwartz et al., 2007)
* Dual Coordinate Descent (Hsieh et al., 2008)

The `svm` function is a wrapper for `pegasos`, but it is
possible to call `cddual` explicitly. See the source code
for the hyperparameters of the `cddual` function.

# Usage

The demo below shows how SVMs work:

```julia
# To show how SVMs work, we'll use Fisher's iris data set
using SVM
using RDatasets

# We'll learn to separate setosa from other species
iris = data("datasets", "iris")

# SVM format expects observations in columns and features in rows
X = array(iris[:, 1:4])'
p, n = size(X)

# SVM format expects positive and negative examples to +1/-1
Y = [species == "setosa" ? 1.0 : -1.0 for species in iris[:, "Species"]]

# Select a subset of the data for training, test on the rest.
train = randbool(n)

# We'll fit a model with all of the default parameters
model = svm(X[:,train], Y[train])

# And now evaluate that model on the testset
accuracy = nnz(predict(model, X[:,~train]) .== Y[~train])/nnz(~train)
```

You may specify non-default values for the various parameters:

```julia
# The algorithm processes minibatches of data of size k
model = svm(X, Y, k = 150)

# Weight regularization is controlled by lambda
model = svm(X, Y, lambda = 0.1)

# The algorithm performs T iterations
model = svm(X, Y, T = 1000)
```

