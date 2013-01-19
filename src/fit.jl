const TRAINBIN = Pkg.dir("SVM", "bin", "svm-train")
const PREDICTBIN = Pkg.dir("SVM", "bin", "svm-predict")

function traincall(datafile::String, modelfile::String)
	run(`$TRAINBIN $datafile $modelfile`)
end

function predictcall(modelfile::String, datafile::String, predictfile::String)
	run(`$PREDICTBIN $datafile $modelfile $predictfile`)
end

# TODO: Make this definition and one below not redundant
function svm(examples::Vector{SVMExample})
	datafile, io = mktemp()
	modelfile = strcat(datafile, ".model")
	write_svm_data(io, examples)
	close(io)
	traincall(datafile, modelfile)
	model = read_svm_model(modelfile)
	rm(datafile)
	rm(modelfile)
	return model
end

function svm{T <: Real, S <: Real}(labels::Vector{S}, features::Matrix{T})
	datafile, io = mktemp()
	modelfile = strcat(datafile, ".model")
	write_svm_data(io, labels, features)
	close(io)
	traincall(datafile, modelfile)
	model = read_svm_model(modelfile)
	rm(datafile)
	rm(modelfile)
	return model
end

# TODO: Is there any way to fix this Union definition to prevent redundancy?
# Problem is probably that T is not bound when using Vector{SVMExample}
function predict{T <: Real}(model::SVMModel, features::Union(Matrix{T}))
	datafile, io = mktemp()
	modelfile = strcat(datafile, ".model")
	predictfile = strcat(datafile, ".predictions")
	write_svm_data(io, features)
	close(io)
	write_svm_model(modelfile, model)
	predictcall(modelfile, datafile, predictfile)
	predictions = float64(split(chomp(readall(predictfile)), r"\s+"))
	rm(datafile)
	rm(modelfile)
	rm(predictfile)
	return predictions
end

function predict(model::SVMModel, features::Vector{SVMExample})
	datafile, io = mktemp()
	modelfile = strcat(datafile, ".model")
	predictfile = strcat(datafile, ".predictions")
	write_svm_data(io, features)
	close(io)
	write_svm_model(modelfile, model)
	predictcall(modelfile, datafile, predictfile)
	predictions = float64(split(chomp(readall(predictfile)), r"\s+"))
	rm(datafile)
	rm(modelfile)
	rm(predictfile)
	return predictions
end
