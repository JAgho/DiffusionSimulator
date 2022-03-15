using Gtk
using DiffusionSimulator, CUDA, StatsBase, Plots, Gtk, MAT, FileIO, JLD2

function main()
fnames = open_dialog("test", select_multiple=true)
res = Dict()
i = 1
sfname = diff_save_interface("output/generic/15px", "15px_shf0_0_s2.jld2")
while i <= length(fnames)
    # try
    @sync begin
    fname = fnames[i]
    println("processing $fname")
    file = matopen(fname)
    I = Int32.(.!read(file, "Im"))
    display(heatmap(I))

    dx = (2/3)*1e-6
    phi = sum(I)/length(I)
    gam=2.675e8
    delta=10e-3
    Delta=10.1e-3
    tf=Delta+delta
    t=collect(range(0, tf, length=1000))
    G=0 .* t
    G[t .<= delta] .= 1
    G[(t .>= Delta) .& (t .<= Delta+delta)] .= -1
    dir = [0,1]
    G = kron(G, dir')

    N = round(100e6/phi)
    seq = Seq(G, t, collect(0:0.001:0.793)) # build seq object
    simu = Simu([1e-9], N, dx, gam) # build sim object

    S = Array(diff_sim_gpu(I, seq, simu))
    res[fname] = S
    end
    i += 1
# catch 
#     res[fname] = "err"
#     println("something has gone wrong...")
# end
end
safe_save(sfname, res)
return res
end
res = main()
#G_s = collect(0:0.01:.5)
#end
#bval=(2.675e8.*G_s).^2*delta^2*(Delta-delta/3);
#bval .*= 1e-9
#plot(bval, collect(values(res))[1], yaxis=:log)
#plot(seq.G_s, abs.(real.(S)), yaxis=:log, ylim=[1e-3, 1])

#JLD2.save("output/generic/results_ordered_c2.jld2", res)


#uh = JLD2.load("results.jld2")
#uh


