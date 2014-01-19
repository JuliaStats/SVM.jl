module SVM

export svm, cddual, pegasos, predict

type SVMFit
	w::Vector{Float64}
	pass::Int
	converged::Bool
end

function Base.show(io::IO, fit::SVMFit)
	@printf io "Fitted linear SVM\n"
	@printf io " * Non-zero weights: %d\n" nnz(fit.w)
	@printf io " * Iterations: %d\n" fit.pass
	@printf io " * Converged: %s\n" string(fit.converged)
end

function predict(fit::SVMFit, X::Matrix)
	n, l = size(X)
	preds = Array(Float64, l)
	for i in 1:l
		tmp = 0.0
		for j in 1:n
			tmp += fit.w[j] * X[j, i]
		end
		preds[i] = sign(tmp)
	end
	return preds
end

include("pegasos.jl")

include("cddual.jl")

include("sparse.jl")

function svm(X::Matrix,
	         Y::Vector;
	         k::Integer = 5,
	         lambda::Real = 0.1,
	         T::Integer = 100)
	pegasos(X, Y, k = k, lambda = lambda, T = T)
end

end # module
