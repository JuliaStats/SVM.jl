using SVM

datafile = joinpath("test", "data", "heart.scale")
altdatafile = strcat(datafile, ".alt")
modelfile = joinpath("test", "data", "heart.scale.model")
altmodelfile = strcat(modelfile, ".alt")
predictfile = joinpath("test", "data", "heart.scale.predict")
altpredictfile = strcat(predictfile, ".alt")

d = SVM.read_svm_data(datafile)
SVM.write_svm_data(altdatafile, d)
# Should we respect ints in data?
# @assert readall(datafile) == readall(altdatafile)
rm(altdatafile)

model = SVM.read_svm_model(modelfile)
SVM.write_svm_model(altmodelfile, model)
# Should we respect ints in data?
# @assert readall(modelfile) == readall(altmodelfile)
rm(altmodelfile)
