function read_svm_example(line::String)
	fields = split(chomp(line), " ")
	label = parse_float(fields[1])
	features = Dict{Int, Float64}()

	for i in 2:(length(fields) - 1)
		feature_index, feature_value = split(fields[i], ":")
		features[int(feature_index)] = parse_float(feature_value)
	end

	return SVMExample(label, features)
end

function write_svm_example(io::IO, e::SVMExample)
	print(io, e.label)
	print(io, ' ')
	# Would be faster as (ind, val)
	# But wouldn't match format of input file
	for ind in sort(keys(e.features))
		print(io, ind)
		print(io, ':')
		print(io, e.features[ind])
		print(io, ' ')
	end
	print(io, '\n')
end

function read_svm_data(io::IO)
	examples = Array(SVMExample, 0)
	for line in each_line(io)
		push!(examples, read_svm_example(line))
	end
	return examples
end

function read_svm_data(pathname::String)
	io = open(pathname, "r")
	examples = read_svm_data(io)
	close(io)
	return examples
end

function write_svm_data(io::IO, examples::Vector{SVMExample})
	for i in 1:length(examples)
		write_svm_example(io, examples[i])
	end
end

function write_svm_data{S <: Real, T <: Real}(io::IO, labels::Vector{S}, features::Matrix{T})
	n, p = size(features)
	for i in 1:n
		write_svm_example(io, SVMExample(labels[i], reshape(features[i, :], p)))
	end
end

function write_svm_data{T <: Real}(io::IO, features::Matrix{T})
	n, p = size(features)
	for i in 1:n
		write_svm_example(io, SVMExample(1, reshape(features[i, :], p)))
	end
end

function write_svm_data(pathname::String, examples::Any)
	io = open(pathname, "w")
	write_svm_data(io, examples)
	close(io)
end

# TODO: Support other orders for entries of model?
function read_svm_model(pathname::String)
	io = open(pathname, "r")
	svm_type = symbol(uppercase(split(chomp(readline(io)), " ")[end]))
	kernel_type = symbol(uppercase(split(chomp(readline(io)), " ")[end]))
	gamma_param = float64(split(chomp(readline(io)), " ")[end])
	num_classes = int(split(chomp(readline(io)), " ")[end])
	total_sv = int(split(chomp(readline(io)), " ")[end])
	rho_param = float64(split(chomp(readline(io)), " ")[end])
	label = int(split(chomp(readline(io)), " ")[2:end])
	nr_sv = int(split(chomp(readline(io)), " ")[2:end])
	@assert chomp(readline(io)) == "SV"
	SV = SVMExample[]
	for line in each_line(io)
		push!(SV, read_svm_example(line))
	end
	return SVMModel(svm_type, kernel_type, gamma_param, num_classes,
		            total_sv, rho_param, label, nr_sv, SV)
end

function write_svm_model(io::IO, m::SVMModel)
	print(io, "svm_type ")
	println(io, lowercase(string(m.svm_type)))

	print(io, "kernel_type ")
	println(io, lowercase(string(m.kernel_type)))

	print(io, "gamma ")
	println(io, m.gamma)

	print(io, "nr_class ")
	println(io, m.nr_class)

	print(io, "total_sv ")
	println(io, m.total_sv)

	print(io, "rho ")
	println(io, m.rho)

	print(io, "label ")
	println(io, join(m.label, " "))

	print(io, "nr_sv ")
	println(io, join(m.nr_sv, " "))

	println(io, "SV")
	for i in 1:length(m.SV)
		write_svm_example(io, m.SV[i])
	end
end

function write_svm_model(pathname::String, m::SVMModel)
	io = open(pathname, "w")
	write_svm_model(io, m)
	close(io)
end
