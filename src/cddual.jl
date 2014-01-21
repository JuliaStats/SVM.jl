# Randomization option slows down processing
# but improves quality of solution considerably
# Would be better to do randomization in place
function cddual(X::Matrix,
	            Y::Vector;
	            C::Real = 1.0,
	            norm::Integer = 2,
	            randomized::Bool = true,
	            maxpasses::Integer = 2)
	# l: # of samples
	# n: # of features
	n, l = size(X)
	alpha = zeros(l)
	w = zeros(n)

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
		Qbar[i] = D[i] + dot(X[:, i], X[:, i])
	end

	# Loop over examples
	converged = false
	pass = 0

	while !converged
		# Assess convergence
		pass += 1
		if pass == maxpasses
			converged = true
		end

		# Choose order of observations to process
		if randomized
			indices = randperm(l)
		else
			indices = 1:l
		end

		# Process all observations
		for i in indices
			g = Y[i] * dot(w, X[:, i]) - 1.0 + D[i] * alpha[i]

			if alpha[i] == 0.0
				pg = min(g, 0.0)
			elseif alpha[i] == U
				pg = max(g, 0.0)
			else
				pg = g
			end

			if abs(pg) > 0.0
				alphabar = alpha[i]
				alpha[i] = min(max(alpha[i] - g / Qbar[i], 0.0), U)
				for j in 1:n
					w[j] = w[j] + (alpha[i] - alphabar) * Y[i] * X[j, i]
				end
			end
		end
	end

	return SVMFit(w, pass, converged)
end
