using SVM

labels = [1, -1, -1, 1, 1, -1]

features = [1.0 0.0;
            0.0 -1.0;
            0.0 -0.9;
            0.9 0.1;
            1.0 1.0;
            -1.0 -1.0;]

# Using Array's

model = svm(labels, features)

predictions = predict(model, features)

# Using SVMExample objects

examples = read_svm_data(joinpath("test", "data", "heart.scale"))

model = svm(examples)

predictions = predict(model, examples)
