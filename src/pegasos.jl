# S is X,Y
# T is maxpasses
# p: # of features
# n: # of data points
# k: size of minibatch
function pegasos{T<:Real}(X::AbstractMatrix{T},
                          Y::AbstractVector{T};
                          k::Integer = 5,
                          lambda::Real = 0.1,
                          maxpasses::Integer = 100)
    # p features, n observations
    p, n = size(X)

    # Initialize weights so norm(w) <= 1 / sqrt(lambda)
    w = randn(p)
    sqrtlambda = sqrt(lambda)
    normalizer = sqrtlambda * norm(w)
    for j in 1:p
        w[j] /= normalizer
    end

    # Allocate storage for repeated used arrays
    deltaw = Array{Float64,1}(p)
    w_tmp = Array{Float64,1}(p)

    # Loop
    for t in 1:maxpasses
        # Calculate stepsize parameters
        alpha = 1.0 / t
        eta_t = 1.0 / (lambda * t)

        # Calculate scaled sum over misclassified examples
        # Subgradient over minibatch of size k
        fill!(deltaw, 0.0)
        for i in 1:k
            # Select a random item from X
            # This is one element of At of S
            index = rand(1:n)

            # Test if prediction isn't sufficiently good
            # If so, current item is element of At+
            pred = Y[index] * dot(w, X[:, index])
            if pred < 1.0
                # Update subgradient
                for j in 1:p
                    deltaw[j] += Y[index] * X[j, index]
                end
            end
        end

        # Rescale subgradient
        for j in 1:p
            deltaw[j] *= (eta_t / k)
        end

        # Calculate tentative weight-update
        for j in 1:p
            w_tmp[j] = (1.0 - alpha) * w[j] + deltaw[j]
        end

        # Find projection of weights into L2 ball
        proj = min(1.0, 1.0 / (sqrtlambda * norm(w_tmp)))
        for j in 1:p
            w[j] = proj * w_tmp[j]
        end
    end

    return SVMFit(w, maxpasses, true)
end
