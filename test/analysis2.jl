using FileIO, JLD2, Plots, StatsBase, Statistics

function decompose_diff_dataset(fname)

    result = JLD2.load(fname)

    meta = []
    
    for key in keys(result)
        f = splitext(basename(key))[1]
        temp = split(f, "l")
        n = parse(Int, temp[1][4:end])
        l = parse(Int, temp[2])
        #println("fname = $f \t n = $n \t l = $l")
        push!(meta, (key, n, l))
    end
    msort = Dict()
    for e in meta
        f = e[1]; n = e[2]; l = e[3]
        #println("fname = $f \t n = $n \t l = $l")
        r = get(msort, l, false)
        if r==false
            msort[l] = result[f]
        else
            msort[l] = hcat(msort[l], result[f])
        end
    end
    return sort!(collect(keys(msort))), msort
end

ord1, msort1 = decompose_diff_dataset("output/generic_dde/antiparallel/15px_disord.jld2")
ord2, msort2 = decompose_diff_dataset("output/generic_dde/antiparallel/15px_disord_c2.jld2")
ord3, msort3 = decompose_diff_dataset("output/generic_dde/antiparallel/15px_disord_s2.jld2")



delta = 10e-3
Delta = 10.1e-3
gs = collect(0:0.001:0.793)
bval=((2.675e8.*gs .* delta).^2).*(Delta-delta/3);
bval .*= 1e-9
plt = plot(yaxis=:log, ylim=(1e-3, 1),xaxis="Gradient field (T/m)")
plot!(plt, bval, mean(msort1[200][:,:], dims=2),  label="S2+C2")
plot!(plt, bval, mean(msort2[200][:,:], dims=2), label="C2")
plot!(plt, bval, mean(msort3[200][:,:], dims=2), label="S2")
S1 = msort1[200][:,:]
S2 = msort2[200][:,:]
S3 = msort3[200][:,:]
m1 = mean(S1, dims=2)
m2 = mean(S2, dims=2)
m3 = mean(S3, dims=2)
covar1 = std(S1, dims=2)./m1
covar2 = std(S2, dims=2)./m2
covar3 = std(S3, dims=2)./m3

b = plot(bval, 
            covar1, 
            xaxis="Gradient field (T/m)", 
            title="Coefficient of variation vs gradient", 
            labels="S2+C2",
            legend=true,
            linecolor=:blue
            )

plot!(b, bval, covar2, label="C2", linecolor=:red)
plot!(b, bval, covar3, label="S2", linecolor=:green)
display(plt)
display(b)
savefig(plt, "output/graphs/disord_signal.png")
savefig(b, "output/graphs/disord_covar.png")