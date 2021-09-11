a = CuArray{Bool}(undef, 100000)
b = CuArray{Float64}(undef, 10000, 2)
b[a]

CUDA.zeros(1)

typeof(b)