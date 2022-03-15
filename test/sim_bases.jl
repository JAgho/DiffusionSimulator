


#include("../src/gpu_workers.jl")
#@everywhere include("../src/sequences.jl")
using JLD2
using CUDA
using DiffusionSimulator

 # mutate filenames to get titles

function load_maps(bdir, fnames, titles)
    Is = Dict()
    for (fname, title) in zip(fnames, titles)
        fname = joinpath(bdir, fname)
        Is[title] = JLD2.load(fname, "I")
    end
    return Is
end
# Is

# for I in values(Is)
#     display(heatmap(I, aspect_ratio=:equal, size=(1000,1000)))
# end


    

function sim_protocol(I::Array{Bool, 2}, Ni::Int64)
    #It = I
    It = .!I
    dx = (2/3)*1e-6
    gam=2.675e8
    phi = sum(It)/length(It)
    N = round(Int64, Ni/phi)
    # build seq object

    simu = Simu([1e-9], N, dx, gam)# build sim object
    #seq = pgse(10e-3, 10.1e-3, 0:0.001:0.793, [0,1], 1000)
    δ = 5e-3
    Δ = 10e-3
    tₘ = 5.01e-3
    #dir = [1/sqrt(2),1/sqrt(2)]
    dir = [1.0,0.0]
    Gₛ = collect(0:0.001:0.793)
    seq = dde(δ, Δ, tₘ, 1000, Gₛ, dir)
    return diff_sim_gpu(It, seq, simu)
end

function batch_sim(I, N)::Tuple{Vector{Matrix{Bool}}, Vector{Int64}}
    nmax = Int64(100e6)
    if N <= nmax
        return [I], [N]
    end

    loops, remainder = round.(Int64, fldmod(N, nmax))
    if remainder > 0 
        r = zeros(Int64,loops+1)
        r[1] = remainder
        r[2:end] .= nmax
        return [I for i in 1:loops+1], r
    else
        r = zeros(Int64,loops)
        r .= nmax
        return [I for i in 1:loops], r
    end
end

function batch_protocol(prot)
    res = []
    for i = 1:length(prot[1])
        push!(res, sim_protocol(prot[1][i], prot[2][i]))
    end
    return res
end

function main()
    bdir = "../YTRecon/bases/16px_200"
    fnames = readdir(bdir)
    titles = map(x->x[6:end-11], readdir(bdir))
    Is = load_maps(bdir, fnames, titles)

    N = 100e6
    Ims = []
    Ns = Int64[]
    lens = []
    for (key, value) in Is
        prot = batch_sim(value, N)
        
        append!(Ims, prot[1])
        append!(Ns, prot[2])
        push!(lens, length(prot[1]))
    end
    res = batch_protocol((Ims, Ns))

    true_res = Dict()
    o = 1
    for (i, k) in zip(lens, keys(Is))
        true_res[k] = res[o:o+i-1]
        o+=i
    end
    return true_res
    #return Ims, Ns, Is

end


res = main()
JLD2.save("output/bases/basis_200_1e8_1.jld2", res)

