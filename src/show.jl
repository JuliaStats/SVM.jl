function show(io::IO, m::SVMModel)
	print(io, "SVM Type: ")
	println(io, lowercase(string(m.svm_type)))

	print(io, "Kernel Type: ")
	println(io, lowercase(string(m.kernel_type)))

	print(io, "Gamma: ")
	println(io, m.gamma)

	print(io, "Number of Classes: ")
	println(io, m.nr_class)

	print(io, "Total Support Vectors: ")
	println(io, m.total_sv)

	print(io, "Rho ")
	println(io, m.rho)

	print(io, "Labels: ")
	println(io, join(m.label, " "))

	print(io, "Numbers of Support Vectors per Label: ")
	println(io, join(m.nr_sv, " "))
end
