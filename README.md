SVM.jl
======

# SVM's in Julia

Native Julia implementations of standard SVM algorithms.
Currently, there are textbook style implementations of
two popular linear SVM algorithms:

* Pegasos (Shalev-Schwartz et al., 2007)
* Dual Coordinate Descent (Hsieh et al., 2008)

The `svm` function is a wrapper for `pegasos`, but it is
possible to call `cddual` explicitly. See the source code
for the hyperparameters of the `cddual` function.

# Usage

The demo below shows how SVM's work:

    # To show how SVM's work, we'll use Fisher's iris data set
    using SVM
    using RDatasets

    # We'll learn to separate setosa from other species
    iris = data("datasets", "iris")

    # SVM format expects observations in columns and features in rows
    X = matrix(iris[:, 2:5])'
    p, n = size(X)

    # SVM format expects positive and negative examples to +1/-1
    Y = Array(Float64, n)
    for i in 1:n
        if iris[i, "Species"] == "setosa"
            Y[i] = 1.0
        else
            Y[i] = -1.0
        end
    end

    # We'll fit a model with all of the default parameters
    model = svm(X, Y)

    # The algorithm processes minibatches of data of size k
    model = svm(X, Y, k = 150)

    # Weight regularization is controlled by lambda
    model = svm(X, Y, lambda = 0.1)

    # The algorithm performs T iterations
    model = svm(X, Y, T = 1000)
