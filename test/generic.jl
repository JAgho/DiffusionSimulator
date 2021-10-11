using Gtk
using DiffusionSimulator, CUDA, StatsBase, Plots, Gtk, MAT, FileIO, JLD2

function main()
fnames = open_dialog("test", select_multiple=true)
res = Dict()
i = 1
fname = fnames[1]
while i <= length(fnames)
    try
    @sync begin
    fname = fnames[i]
    println("processing $fname")
    file = matopen(fname)
    I = Int32.(.!read(file, "Im"))
    #display(heatmap(I, ratio=:equal))
    #ra=10e-6; # [m]
    #N_ii=60; # number of side pixels
    #len = collect(range(-ra*1.3, ra*1.3, length=N_ii))
    dx = 1e-6#abs(len[2]-len[1])

    phi = sum(I)/length(I)
    gam=2.675e8

    # Setup sequence
    delta=5e-3;    Delta=50e-3;    tf=2*Delta
    t=collect(range(0, tf, length=1000))
    G=0 .* t
    G[t .<= delta] .= 1
    G[(t .>= Delta) .& (t .<= Delta+delta)] .= -1
    dir = [0,1]
    G = kron(G, dir')
    dt=t[2]-t[1]

    N = round(10e6/phi)
    seq = Seq(G, t, collect(0:0.01:.5)) # build seq object
    simu = Simu([1e-9], N, dx) # build sim object

    phase = diff_sim_gpu(I, seq, simu)
    phasec = zeros(Float64, length(phase))
    copyto!(phasec, phase)
    comp = phasec .* (0-1im)


    S=zeros(ComplexF64, length(seq.G_s))
    for gg in 1:length(S); S[gg]=mean(exp.(comp*gam*dt*seq.G_s[gg])); end
    res[fname] = S
    end
    i += 1
catch 
    res[fname] = "err"
end
end
return res
end
res = main()
#plot(seq.G_s, abs.(real.(S)), yaxis=:log, ylim=[1e-3, 1])

JLD2.save("results_disordered.jld2", res)
#uh = JLD2.load("results.jld2")
#uh

