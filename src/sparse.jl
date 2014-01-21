# Exploit sparsity in w and X
# Extraordinarily slower
function svm2(X::Matrix, #SparseMatrixCSC,
	         Y::Vector;
	         C::Real = 0.1,
	         norm::Integer = 2,
	         randomized::Bool = true,
	         maxpasses::Integer = 5)
	n, l = size(X)
	alpha = zeros(l)
	w = zeros(n) # Should be spzeros as vector

	# Set U and D
	#  * L1-SVM: U = C, D[i] = 0
	#  * L2-SVM: U = Inf, D[i] = 1 / (2C)
	U = 0.0
	D = Array(Float64, l)
	if norm == 1
		U = C
		for i in 1:l
			D[i] = 0.0
		end
	elseif norm	== 2
		U = Inf
		for i in 1:l
			D[i] = 1.0 / (2.0 * C)
		end
	else
		DomainError("Only L1-SVM and L2-SVM are supported")
	end

	# Set Qbar
	Qbar = Array(Float64, l)
	for i in 1:l
		Qbar[i] = D[i]
		for k in 1:n
			Qbar[i] += (X[:, i]'X[:, i])[1]
		end
	end

	# Loop over examples
	converged = false
	pass = 0

	while !converged
		pass += 1
		if pass == maxpasses
			converged = true
		end
		if randomized
			# This slows things down a lot, but improves
			#  model fit quality considerably
			indices = randperm(l)
		else
			indices = 1:l
		end
		for i in indices
			G = Y[i] * (w'X[:, i])[1] - 1.0 + D[i] * alpha[i]

			if alpha[i] == 0.0
				PG = min(G, 0.0)
			elseif alpha[i] == U
				PG = max(G, 0.0)
			else
				PG = G
			end

			if abs(PG) > 0.0
				alphabar = alpha[i]
				alpha[i] = min(max(alpha[i] - G / Qbar[i], 0.0), U)
				w = w + (alpha[i] - alphabar) * Y[i] * X[:, i]
			end
		end
	end

	return SVMFit(w, pass, converged)
end
