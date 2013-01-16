module SVM
	import Base.show

	export svm, predict
	export SVMExample, SVMModel

	include("model.jl")
	include("io.jl")
	include("fit.jl")
	include("show.jl")
end
