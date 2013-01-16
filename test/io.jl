using SVM

datafile = joinpath("test", "data", "heart.scale")
altdatafile = strcat(datafile, ".alt")
modelfile = joinpath("test", "data", "heart.scale.model")
altmodelfile = strcat(modelfile, ".alt")
predictfile = joinpath("test", "data", "heart.scale.predict")
altpredictfile = strcat(predictfile, ".alt")

d = read_svm_data(datafile)
write_svm_data(altdatafile, d)
# Should we respect ints in data?
# @assert readall(datafile) == readall(altdatafile)
rm(altdatafile)

model = read_svm_model(modelfile)
write_svm_model(altmodelfile, model)
# Should we respect ints in data?
# @assert readall(modelfile) == readall(altmodelfile)
rm(altmodelfile)
