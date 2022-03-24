using JLD2, MAT

data = JLD2.load("output/16px/disord.jld2")
bases = JLD2.load("output/bases/basis_500_1e8_final.jld2")
matwrite("scratch/disord.mat", data)
matwrite("scratch/bases.mat", data)

δ = 5e-3
Δ = 10e-3
tₘ = 5.1e-3
#dir = [1/sqrt(2),1/sqrt(2)]
dir = [1.0,0.0]
Gₛ = collect(0:0.001:0.793)
seq = dde(δ, Δ, tₘ, 1000, Gₛ, dir)
seq2 = pgse(10e-3, 10.1e-3, 0:0.001:0.793, [0,1], 1000)
seq.Gray
plot(seq.G)
sum(seq.G)
sum(seq2.G)