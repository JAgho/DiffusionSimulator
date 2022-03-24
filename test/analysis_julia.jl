using Plots, JLD2, StatsBase, Statistics

function agglomerate_results()
    Is = []
    for i in 1:6
        push!(Is, JLD2.load("output/16px/bases_ex$i.jld2"))
    end

    k = keys(Is[1])
    v = collect(values(Is[1]))
    res = Dict()
    for (i, key) in enumerate(k)
        res[key] = v[i]
    end


    for key in k
        for group in Is[2:end]
            res[key] = hcat(res[key], group[key])
        end
    end

    for key in k
        res[key] = mean(res[key], dims=2)
    end

    res
end

function load_diff_sim(fname)
    result = JLD2.load(fname)

    res = result["result"]
    l = length(res[1])
    n = length(res)
    s = zeros(Float64, l, n)
    for i in 1:n
        s[:, i] .= res[i]
    end
    return s
end

function load_pair(fname)
    f1 = fname 
    f2 = splitext(fname)
    f2 = f2[1]*"_sq"*f2[2]
    (load_diff_sim(f1), load_diff_sim(f2))
end

#s = load_diff_sim("output/16px/disord.jld2")


function plot_pair(bval, re, bname; m=true)
    if m==true
        plt = plot(yaxis=:log, ylim=(1e-4, 1),xaxis="b-value (s/um²)", title = bname*" S(q)+cS(q) signal", size = (1000,1000))
        plot!(plt, bval, mean(re[1], dims=2),  label="mean sq+csq", lc=:red, linewidth=3, linestyle=:dash)
        plot!(plt, bval, mean(re[2], dims=2),  label="mean sq", lc=:black, linewidth=3, linestyle=:dash)
        return plt
    else
        plt = plot(yaxis=:log, ylim=(1e-4, 1),xaxis="b-value (s/um²)", title = bname*" S(q)+cS(q) signal", size = (1000,1000))
        plot!(plt, bval, mean(re[1], dims=2), ribbon = std(re[1], dims=2),  label="mean sq+csq", lc=:red, linewidth=3, linestyle=:dash)
        #plot!(plt, bval, re[1])
        plot!(plt, bval, mean(re[2], dims=2), ribbon = std(re[2], dims=2), label="mean sq", lc=:black, linewidth=3, linestyle=:dash)
        #plot!(plt, bval, re[2])
    return plt
    end
end

function covar_pair(bval, re, bname)
    m1 = mean(re[1], dims=2)
    covar1 = std(re[1], dims=2)./m1
    m2 = mean(re[2], dims=2)
    covar2 = std(re[2], dims=2)./m2
    plt = plot(xaxis="b-value (s/um²)", title = bname*" S(q)+cS(q) Coefficient of Variation", size = (1000,1000))
    plot!(plt, bval, covar1,  label="mean sq+csq")
    plot!(plt, bval, covar2,  label="mean sq")
    return plt
end

function analyse_pair(fname)
    delta = 10e-3
    Delta = 10.1e-3
    gs = collect(0:0.001:0.793)
    bval=((2.675e8.*gs .* delta).^2).*(Delta-delta/3);
    bval .*= 1e-9
    re = load_pair(fname)
    bn = basename(fname)
    display(plot_pair(gs, re, bn))
    display(plot_pair(gs, re, bn, m=false))
    display(covar_pair(gs, re, bn))
end

fs = ["disord.jld2", "ord.jld2", "disordel.jld2", "ordel.jld2"]
fr = ["shuf_1.jld2","shuf_2.jld2", "shuf_3.jld2", "shuf_4.jld2", "shuf_5.jld2"]
ft = ["disordel.jld2", "ordel.jld2"]
# for i in fs
#     analyse_pair("output/16px/"*i)
# end


#plt = plot(yaxis=:log, ylim=(1e-3, 1),xaxis="b-value (s/um²)", title = " S(q)+cS(q) signal", size = (1500,1000))
#bases = agglomerate_results()

bases = JLD2.load("output/bases/basis_500_dde_final.jld2")


for i in append!(fs, fr)
    
    fname = "output/16px_dde_2/"*i
    k = i[1:end-5]
    plt = plot(yaxis=:log, ylim=(1e-3, 1),xaxis="Max Field (T/m)", title = " S(q)+cS(q) signal "*k, size = (1500,1000))
    delta = 10e-3
    Delta = 10.1e-3
    gs = collect(0:0.001:0.793)
    bval=((2.675e8.*gs .* delta).^2).*(Delta-delta/3);
    bval .*= 1e-9
    re = load_pair(fname)
    ip = i[1:end-5]
    plot!(plt, gs, mean(re[1], dims=2), ribbon = std(re[1], dims=2), label=ip*" mean sq+csq")
    plot!(plt, gs, mean(re[2], dims=2), ribbon = std(re[2], dims=2), label=ip*" mean sq")
    plot!(plt, gs, bases[k], label=ip*" basis", linestyle=:dash)
    display(plt)
end




