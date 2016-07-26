SVM.jl
======

# THIS PACKAGE IS UNMAINTAINED AND WILL BE REMOVED FROM METADATA

[![Project Status: Unsupported - The project has reached a stable, usable state but the author(s) have ceased all work on it. A new maintainer may be desired.](http://www.repostatus.org/badges/latest/unsupported.svg)](http://www.repostatus.org/#unsupported)
[![Build Status](https://travis-ci.org/JuliaStats/SVM.jl.svg?branch=master)](https://travis-ci.org/JuliaStats/SVM.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaStats/SVM.jl/badge.svg)](https://coveralls.io/r/JuliaStats/SVM.jl)
[![SVM](http://pkg.julialang.org/badges/SVM_0.4.svg)](http://pkg.julialang.org/?pkg=SVM&ver=0.4)


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
iris = dataset("datasets", "iris")

# SVM format expects observations in columns and features in rows
X = array(iris[:, 1:4])'
p, n = size(X)

# SVM format expects positive and negative examples to +1/-1
Y = [species == "setosa" ? 1.0 : -1.0 for species in iris[:Species]]

# Select a subset of the data for training, test on the rest.
train = randbool(n)

# We'll fit a model with all of the default parameters
model = svm(X[:,train], Y[train])

# And now evaluate that model on the testset
accuracy = countnz(predict(model, X[:,~train]) .== Y[~train])/countnz(~train)
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

