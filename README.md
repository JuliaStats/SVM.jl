SVM.jl
======

# About

An interface to libSVM from Julia

# Installation

* Go into `deps`
* Run `make`
* Run `cp svm-train ../bin`
* Run `cp svm-predict ../bin`

# Usage

    using SVM

    labels = [1, -1, -1, 1, 1, -1]

    features = [1.0 0.0;
                0.0 -1.0;
                0.0 -0.9;
                0.9 0.1;
                1.0 1.0;
                -1.0 -1.0;]

    model = svm(labels, features)

    predictions = predict(model, features)

    using RDatasets

    iris = data("datasets", "iris")

    labels = vector(iris["Species"])
    labels = int(labels .== "setosa")

    features = matrix(iris[:, 2:5])

    model = svm(labels, features)
    predictions = predict(model, features)
    mean(predictions .== labels)

# To Do

* Use binary interface to libSVM instead of libSVM text format
* Convert sparse matrices to libSVM format
* Restore `Makefile.win` and `windows` in `deps`
* Allow users to configure SVM model fitting process
