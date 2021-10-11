using FileIO, JLD2, Plots, StatsBase, Statistics

result = JLD2.load("results_disordered.jld2")
ideal = JLD2.load("results_ideal.jld2")
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
    #println("fname = $f \t n = $n \t l = $l")
    r = get(msort, l, false)
    if r==false
        msort[l] = result[f]
    else
        msort[l] = hcat(msort[l], result[f])
    end
end
ord = sort!(collect(keys(msort)))
ana = ideal["analytical"]
errors = []
stds = []
for k in ord
    gs = "Gradient field (T/m)"
    S = abs.(real.(msort[k]))
    #println(size(msort[k]), " \t$k")
    a = plot(G_s, abs.(real.(ana)), xaxis=gs, title="Combined results", legend=false)
    plot!(G_s, S, yaxis=:log)
    err = abs.(mean(S .- ana, dims = 2))
    push!(errors, sum(err))
    #dump(err)
    b = plot(G_s, err, xaxis=gs, title="Absolute error vs gradient for l = $k", legend=false)

    stdv = std(S, dims=2)
    c = plot(G_s, stdv, xaxis=gs, title="σ vs gradient for l = $k", legend=false)
    push!(stds, sum(stdv))
    #display(a)
    #display(b)
    #display(c)
    w = plot(a,b,c, layout=(3,1), size=(1000,800))
    display(w)
end

d = scatter(ord, errors ./ 51, xaxis = "Linear dimension", title="Average error vs image linear dimensions", legend = false)
e = scatter(ord, stds, xaxis = "Linear dimension", title="Sum of σ vs image linear dimensions", legend=false)
display(d)
display(e)





