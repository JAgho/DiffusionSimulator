using FileIO, JLD2, Plots, StatsBase

result = JLD2.load("result2.jld2")
meta = []
G_s = collect(0:0.01:.5)
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
    println("fname = $f \t n = $n \t l = $l")
    r = get(msort, l, false)
    if r==false
        msort[l] = result[f]
    else
        msort[l] = hcat(msort[l], result[f])
    end
end
ord = sort!(collect(keys(msort)))
for k in ord
    println(size(msort[k]), " \t$k")
    display(plot(G_s, abs.(real.(msort[k])), yaxis=:log))
end



