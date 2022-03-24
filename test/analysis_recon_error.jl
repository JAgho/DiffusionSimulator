using Plots, JLD2, Statistics

ldir = "output/recon_error/"
fnames = readdir(ldir)
names = sortperm([parse(Float64, i[8:end-8]) for i in fnames], rev=true)
spaths = map(x->joinpath(ldir, x), fnames[names])
results = map(x->hcat(x...), JLD2.load.(spaths, "result"))


begin
plt = plot(yaxis=:log, ylim=(1e-3, 1),xaxis="Max Field (T/m)", title = " S(q)+cS(q) signal ", size = (1500,1000))
for (res, name) in zip(results, fnames[names])
    
    delta = 10e-3
    Delta = 10.1e-3
    gs = collect(0:0.001:0.793)
    bval=((2.675e8.*gs .* delta).^2).*(Delta-delta/3);

    plot!(plt, gs, mean(res, dims=2), ribbon = std(res, dims=2), label=name)
    #plot!(plt, gs, mean(re[2], dims=2), ribbon = std(re[2], dims=2), label=ip*" mean sq")
    #plot!(plt, gs, bases[k], label=ip*" basis", linestyle=:dash)
    
end

display(plt)
end