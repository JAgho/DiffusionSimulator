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

ord1, msort1 = decompose_diff_dataset("output/generic/15px/15px_ord_greb.jld2")
ord2, msort2 = decompose_diff_dataset("output/generic/15px/15px_ord.jld2")
#ord3, msort3 = decompose_diff_dataset("output/generic/results_disordered_c2.jld2")
ord = intersect(ord1, ord2)
ideal = JLD2.load("output/generic/results_ideal.jld2")
ana = ideal["analytical"]
errors = []
v1 = []
v2 = []
v3 = []
diffs = []

G_s = collect(0:0.01:.5)
for k in ord
    l = @layout [a1; b1 c1]
    gs = "Gradient field (T/m)"
    S1 = abs.(real.(msort1[k]))
    S2 = abs.(real.(msort2[k]))
    #S3 = abs.(real.(msort3[k]))
    #println(size(msort[k]), " \t$k")
    a = plot(G_s, 
            abs.(real.(S1))[:,1],
            xaxis=gs, 
            title="Reconstructions with l = $k", 
            legend=true, 
            label="S2+C2",
            linecolor = :blue,
            yaxis=:log)
    plot!(a, G_s, abs.(real.(S2))[:,1], linecolor=:red, label="S2")
    #plot!(a, G_s, abs.(real.(S3))[:,1], linecolor=:orange, label="C2")
    for i = 1:size(S1)[2] - 1
        plot!(a,G_s, abs.(real.(S1))[:,i], label="" , lc=:blue)
        plot!(a,G_s, abs.(real.(S2))[:,i],  label="", lc=:red)
        #plot!(a,G_s, abs.(real.(S3))[:,i],  label="", lc=:orange)
    end
    
    #plot!(G_s, abs.(real.(ana)), label="Grebenkov ideal result", lc=:green)
    #err = abs.(mean(S .- ana, dims = 2))
    #push!(errors, sum(err))
    #dump(err)
    #b = plot(G_s, ana, xaxis=gs, title="Absolute error vs gradient for l = $k", legend=false)
    m1 = mean(S1, dims=2)
    m2 = mean(S2, dims=2)
    #m3 = mean(S3, dims=2)
    covar1 = std(S1, dims=2)./m1
    covar2 = std(S2, dims=2)./m2
    #covar3 = std(S3, dims=2)./m3

    b = plot(G_s, 
                covar1, 
                xaxis=gs, 
                title="Coefficient of variation vs gradient", 
                labels="S2+C2",
                legend=true,
                linecolor=:blue
                )

    plot!(b, G_s, covar2, label="S2", linecolor=:red)            
    #plot!(b, G_s, covar3, label="C2", linecolor=:orange)

    push!(v1, sum(covar1))
    #push!(v2, sum(covar2))
    #push!(v3, sum(covar3))
    c = plot(G_s, abs.(m2.-m1), title="Difference of means", legend=false)
    push!(diffs, abs.(m2.-m1))
    #display(a)
    #display(b)
    #display(c)
    w = plot(a,b, c, layout=l, size=(1000,800))
    display(w)

    savefig(w, "output/res_"*"$k"*"_dis.png")
end

#d = scatter(ord, errors ./ 51, xaxis = "Linear dimension", title="Average error vs image linear dimensions", legend = false)
e = scatter(ord, v1, xaxis = "Linear dimension", 
            title="Coefficient of variation vs image linear dimensions", 
            label = "S2+C2",
            mc=:blue,
            legend=true)
scatter!(e, ord, v2, label = "S2", mc=:red)
#scatter!(e, ord, v3, label = "C2", mc=:orange)
#display(d)
savefig(e, "output/covar_dis.png")
display(e)





