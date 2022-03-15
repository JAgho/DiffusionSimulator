using Gtk
using DiffusionSimulator, CUDA, StatsBase, Plots, Gtk, MAT, FileIO, JLD2

function dde(δ₁, Δ₁, δ₂, Δ₂, tₘ, n, Gₛ, dir)
    tf = Δ₁ + Δ₂ + tₘ + δ₂
    t = collect(range(0, tf, length=1000))
    G=0 .* t
    G[(t .>= 0) .& (t .<= δ₁)] .= -1
    G[(t .>= Δ₁) .& (t .<= Δ₁+δ₁)] .= 1
    G[(t .>= Δ₁+tₘ) .& (t .<= Δ₁+tₘ+δ₂)] .= 1
    G[(t .>= Δ₁+tₘ+Δ₂) .& (t .<= Δ₁+tₘ+Δ₂+δ₂)] .= -1
    G = kron(G, dir')
    Seq(G, t, Gₛ)
end



function dde(δ, Δ, tₘ, n, Gₛ, dir)
    return dde(δ, Δ, δ, Δ, tₘ, n, Gₛ, dir)
end


function main()
fnames = open_dialog("test", select_multiple=true)
res = Dict()
i = 1
sfname = diff_save_interface("output/generic_dde/antiparallel/", "15px_disord_c2.jld2")
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
    #delta=10e-3
    #Delta=10.1e-3
    #tf=Delta+delta
    #tf = 15e-3
    δ = 5e-3
    Δ = 10e-3
    tₘ = 5.01e-3
    dir = [0,1]
    Gₛ = collect(0:0.001:0.793)
    seq = dde(δ, Δ, tₘ, 1000, Gₛ, dir)

    #t=collect(range(0, tf, length=1000))

    #G=0 .* t
    #G[t .<= delta] .= 1
    #G[(t .>= Delta) .& (t .<= Delta+delta)] .= -1
    
    #G = kron(G, dir')
    #seq = Seq(G, t, collect(0:0.001:0.793))
    N = round(10e6/phi)
     # build seq object
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


