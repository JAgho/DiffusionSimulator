using FileIO, JLD2, Plots, StatsBase, Statistics, MAT

function diff_dataset(fname)
    result = JLD2.load(relpath(fname, pwd()))
        return vals = first(values(result))

end

function decompose_diff_dataset(fname)
    result = JLD2.load(relpath(fname, pwd()))
        return values(result)
end

function return_keys(fname)
    result = JLD2.load(fname)
end

#/home/c1869192/srf/SRF/Bases/15px/shuffled
fnames = ["output/generic/15px/15px_shf1_s2.jld2"
"output/generic/15px/15px_shf0_8_s2.jld2"
"output/generic/15px/15px_shf0_5_s2.jld2"
"output/generic/15px/15px_shf0_3_s2.jld2"
"output/generic/15px/15px_shf0_0_s2.jld2"
]
results = diff_dataset.(fnames)
res2 = decompose_diff_dataset.(fnames)
everything = return_keys.(fnames)



tags = [
    "0.5rₐ"
    "0.4rₐ"
    "0.25rₐ"
    "0.15rₐ"
    "0.0rₐ"
]

delta = 10e-3
Delta = 10.1e-3
gs = collect(0:0.001:0.793)
bval=((2.675e8.*gs .* delta).^2).*(Delta-delta/3);
bval .*= 1e-9
a = values(res2)

t = [[] for i in 1:5]
for (i, b) in enumerate(a)
    t[i] = collect(values(b))
end

begin
    plt = [plot(yaxis=:log, ylim=(1e-3, 1),xaxis="Gradient field (T/m)", size=(800,500)) for i in 1:5]

    for (i, entry) in enumerate(t)
        plot!(plt[i], bval, entry,  label=tags[i])
    end

    for plot in plt
    display(plot)
    end
end
# function get_mat(ev)
#     a = relpath(".", first(keys(ev)))
#     b = joinpath(a, first(keys(ev)))
#     println(b)
#     file = matopen(b)
#     I = Int32.(read(file, "Im"))
#     #S2 = Float32.(read(file, "S2"))
#     #C2 = Float32.(read(file, "C2")[end])
#     return I#, S2, C2
# end
# # cd("srf/SRF/Bases/15px/shuffled/shuf1")

# sds = get_mat.(everything)
# for (i,entry) in enumerate(sds)
#     p1 = heatmap(entry[1], aspect_ratio=:equal)
#     #p2 = heatmap(entry[2], aspect_ratio=:equal)
#     #p3 = heatmap(entry[3], aspect_ratio=:equal)
#     display(heatmap(p1,title=tags[i]))
#     #display(heatmap(p2,p3,title=tags[i] , layout=(1,2)))
# end


a = relpath(".", "home/c1869192/srf/SRF/Bases/15px/shuffled/shuf1/sb_200.mat")
b = joinpath(a, "home/c1869192/srf/SRF/Bases/15px/shuffled/shuf1/sb_200.mat")

### Radial distribution statistics ###
#TODO Add ability to compute the mean across radii from the centre of an image

I = zeros(61,61)
function radial_average(I)
    dims = size(I)
    mid = Tuple(round.(Int64, ([dims...].+1) ./ 2))
    shape = CartesianIndices(dims)
    rvals = shape .- CartesianIndex(mid)
    rs = zeros(size(I))
    norm(x,y) = sqrt(x*x + y*y)
    for ((i,j), idx) in zip(Tuple.(rvals), shape)
        rs[idx] = norm(i,j)
        
    end
    r_max = maximum(rs)
    bins = collect(0:1.0:ceil(r_max))
    histo = zeros(Int64, length(bins))
    av = zeros(Float64, length(bins))
    nbins = length(bins)
    rb = (nbins)/r_max 
    for idx in shape
        rIndex=floor(Int, rs[idx]*rb) +1
        if rIndex < nbins+1
            (histo[rIndex] += 1)
            (av[rIndex] += I[idx])
        end
        #bins[fld(Int64, rs[idx], 1.0)] += 1
    end
    av ./=  histo
    return (bins, av)
end

for (i,entry) in enumerate(sds)
    bins1, ra1 = radial_average(entry[2])
    bins2, ra2 = radial_average(entry[3])
    a1 = plot(bins1, ra1, xaxis="radial distance", yaxis="intensity", title = tags[i]*" S2")
    a2 = plot(bins2, ra2, xaxis="radial distance", yaxis="intensity", title = tags[i]*" C2")
    display(a1)
    display(a2)
end

I = rand(61,61)
radial_average(I)


