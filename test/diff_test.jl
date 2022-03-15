using DiffusionSimulator, CUDA, JLD2, Images, Plots
include("../src/sequences.jl")
#println("devices are: ", CUDA.devices())
#println("trying to print devices")
#sleep(10)
CUDA.device!(2)

dirs = ["../YTRecon/bases/16px_2", "../YTRecon/bases/16px_1000"]
fs = ["16px_ord_basis.jld2", "16px_ord_basis.jld2"]

Is = []
for i in 1:2
    push!(Is, JLD2.load(joinpath(dirs[i], fs[i]), "I"))
end


#compose sequence


#build simulation
Ss = []
gs = 0:0.001:0.793
for (i,I) in enumerate(Is)
    dx = (2/3)*1e-6
    gam=2.675e8
    phi = sum(I)/length(I)
    N = round(100e6/phi)
        # build seq object

    simu = Simu([1e-9], N, dx, gam) # build sim object
    seq = pgse(10e-3, 10.1e-3, gs, [0,1], 1000)
    # δ = 5e-3
    # Δ = 10e-3
    # tₘ = 5.01e-3
    # dir = [0,1]
    # Gₛ = collect(0:0.001:0.793)
    # seq = dde(δ, Δ, tₘ, 1000, Gₛ, dir)
    #run simulation
    S = Array(diff_sim_gpu(.!Is[i], seq, simu))
    push!(Ss, S)
end

d = Dict{String, Array{Float64,2}}()
for (S, fname) in zip(Ss, fs)
    d[fname] = S
end

JLD2.save("test.jld2", d)


plt = plot(size=(1000,1000), xlabel="Max B field (T/m)", yaxis=:log, ylim=(1e-3, 1), title = "ordered, extracellular")
plot!(plt, gs, Ss[1], label = "500x500, 1e8 walkers")
plot!(plt, gs, Ss[2], label = "1000x1000, 1e8 walkers")
