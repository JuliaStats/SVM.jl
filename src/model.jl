const SVM_TYPES = [:C_SVC, :NU_SVC, :ONE_CLASS, :EPSILON_SVR, :NU_SVR]
const KERNEL_TYPES = [:LINEAR, :POLY, :RBF, :SIGMOID, :PRECOMPUTED]

type SVMExample
	label::Float64
	features::Dict{Int, Float64}
end

function SVMExample{T <: Real}(label::Real, features::Vector{T})
	feature_dict = Dict{Int, Float64}()
	for i in 1:length(features)
		feature_dict[i] = features[i]
	end
	return SVMExample(float64(label), feature_dict)
end

type SVMModel
	svm_type::Symbol # A value from SVM_TYPES
	kernel_type::Symbol # A value from KERNEL_TYPES
	gamma::Float64 # Parameter
	nr_class::Int # Number of classes
	total_sv::Int # Number of Support Vectors
	rho::Float64 # Parameter
	label::Vector{Int} # Vector of possible labels
	nr_sv::Vector{Int} # Number of Support Vectors for each class
	SV::Vector{SVMExample}
end
