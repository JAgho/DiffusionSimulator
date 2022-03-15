using DiffusionSimulator, CUDA, StatsBase, Plots, Gtk, JLD2

sfname = diff_save_interface("output/greb", "gvar_ex5e6x100.jld2")

ra=5e-6; # [m]
res = Dict()
split = "h"#"h"#"v"
for ra in [5e-6, 10e-6, 15e-6, 19e-6]
    N_ii=100; # number of side pixels
    #len = collect(range(-ra*1.3, ra*1.3, length=N_ii))
    len = collect(range(-20e-6, 20e-6, length=N_ii))
    dgrid = zeros(N_ii, N_ii)
    for i in 1:N_ii
        for j in 1:N_ii
            dgrid[i,j] = sqrt((len[i]^2) + len[j]^2)
        end
    end
    I = zeros(Int32, size(dgrid))
    I[dgrid .>= ra] .= 1
    mindex = map(x-> floor(Int64, x / 2), size(I))
    println(mindex)
    I2 = similar(I)
    
    if split == "h"
        m = mindex[1]
        I2[1:m, :] .= I[m+1:end, :]
        I2[m+1:end, :] .= I[1:m, :]
    elseif split == "v"
        m = mindex[2]
        I2[:, 1:m] .= I[:, m+1:end]
        I2[:, m+1:end] .= I[:, 1:m]
    else 
        I2 .= I
    end
    display(heatmap(I2))
    I .= I2
    phi = sum(I)/length(I)
    gam=2.675e8

    # Setup sequence
    delta=10e-3
    Delta=10.1e-3
    tf=Delta+delta
    t=collect(range(0, tf, length=1000))
    G= 0 .* t
    dt=t[2]-t[1]
    G[t .<= delta] .= 1
    G[(t .>= Delta) .& (t .<= Delta+delta)] .= -1
    dir = [1,1]./sqrt(2)
    G = kron(G, dir')

    N = Float32(round(10e6/phi))
    seq = Seq(G, t, collect(0:0.001:0.793)) # build seq object
    simu = Simu([1e-9], N, abs(len[2]-len[1]), gam) # build sim object
    println("dx = ", abs(len[2]-len[1]))

    S = Array(diff_sim_gpu(I, seq, simu))


    bval=((2.675e8.*seq.G_s .* delta).^2).*(Delta-delta/3);
    bval .*= 1e-9
    display(plot(bval, abs.(real.(S)), yaxis=:log))
    key = "um"*string(Int(ra*1e6))
    res[key] = S
end



safe_save(sfname, res)




#Int32(2.195122e9)