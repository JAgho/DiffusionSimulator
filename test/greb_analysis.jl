using FileIO, JLD2, Plots, StatsBase, Statistics, MAT, Optim

gname1 = "greb.mat"
gname2 = "greb_res.mat"
fname1 = "output/greb/gvar_ex5e6x100.jld2"
fname2 = "output/greb/gvar_ex5e7x100.jld2"
greb = read(matopen(gname1))
grebana = read(matopen(gname2))
gres = (grebana["res"])
gr = Dict()
k = collect(keys(greb))
for i in 1:length(k)
    str = k[i]
    print(str)
    gr[str] = vec(gres[i, 1])
end

gbv = vec(grebana["bval"])
num1 = JLD2.load(fname1)
num2 = JLD2.load(fname2)

delta = 10e-3
Delta = 10.1e-3
gs = collect(0:0.001:0.793)
bval=((2.675e8.*gs .* delta).^2).*(Delta-delta/3);
bval .*= 1e-9

a = plot(yaxis=:log, ylim=(1e-3, 1), 
    title="Grebenkov analytical results vs numerical simulations",
    xaxis = "b (ms/μm²)",
    )
for (key, value) in gr
    #print(value)
    bv = value[:,1]
    S = abs.(real.(value))
    lab = key[3:end]*"μm"
    Sn1 = num1[key]
    Sn2 = num2[key]
    plot!(a, greb[key][:,1], greb[key][:,2], label=lab*" greb result", ls=:dash, lc=:green)
    plot!(a, bval, abs.(real.(Sn1)), label=lab*" numerical", lc=:blue)
    plot!(a, bval, abs.(real.(Sn2)), label=lab*" split num.", lc=:red)
end


#plot!(a, bval, num["numerical"], label="numerical")

display(a)
savefig(a, "output/graphs/grebenkov.png")


S(g, γ, Δ, δ, D) = exp(-(γ^2)*(g^2)*(δ^2)*D*(Δ - δ/3))

S(b, D) = exp(-b*D)

function sqerror(D, b, s)
    err = 0.0
    for i in 1:length(s)
        pred_i = S(b[i], D)
        err += (s[i] - pred_i)^2
    end
    return err
end


begin
r = plot(yaxis=:log, ylim=(1e-3, 1), legend=true, annotationfontsize=6, annotationhalign=:center)
Ds = []
for k in keys(gr)
    len = length(greb[k][:,1])
    x = greb[k][:,1].*1e9
    y = greb[k][:,2]
    result = optimize(D -> sqerror(first(D), x, y), [1.1e-9], LBFGS(), autodiff = :forward)
    result.minimizer

    plot!(r, x, S.(x, result.minimizer), lc=:red, label = "optimised solution")
    plot!(r, x, y, lc=:blue, label = "grebenkov")
    push!(Ds, result.minimizer)
end
eh = ""
for (Dv, k) in zip(Ds, keys(gr))
    ds = @sprintf("%0.3e", first(Dv))
    siz = "$k"
    siz = siz[3:end]*"μm"
    eh = eh*"D = "*ds*" for $siz\n"
end
annotate!((1e10,0.5, eh))
display(r)
end
