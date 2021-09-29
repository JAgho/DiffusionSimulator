using DiffusionSimulator, CUDA, StatsBase, Plots

    ra=10e-6; # [m]
    N_ii=60; # number of side pixels
    len = collect(range(-ra*1.3, ra*1.3, length=N_ii))
    dgrid = zeros(N_ii, N_ii)
    for i in 1:N_ii
        for j in 1:N_ii
            dgrid[i,j] = sqrt((len[i]^2) + len[j]^2)
        end
    end
    I = zeros(Int32, size(dgrid))
    I[dgrid .>= ra] .= 1
    phi = sum(I)/length(I)
    gam=2.675e8

    # Setup sequence
    delta=5e-3
    Delta=50e-3
    tf=2*Delta
    t=collect(range(0, tf, length=1000))
    G=0 .* t
    dt=t[2]-t[1]
    G[t .<= delta] .= 1
    G[(t .>= Delta) .& (t .<= Delta+delta)] .= -1
    dir = [0,1]
    G = kron(G, dir')

    N = round(1e6/phi)
    seq = Seq(G, t, collect(0:0.01:.5)) # build seq object
    simu = Simu([2.3e-9], N, abs(len[2]-len[1])) # build sim object

#CUDA.@time diff_sim_gpu(I, seq, simu)
phase = diff_sim_gpu(I, seq, simu)
phasec = zeros(Float64, length(phase))
copyto!(phasec, phase)
comp = phasec .* (0-1im)
mean(phasec)

S=zeros(ComplexF64, length(seq.G_s))
#exp.(comp*gam*dt*seq.G_s[10])
for gg in 1:length(S)
    S[gg]=mean(exp.(comp*gam*dt*seq.G_s[gg]))
end

plot(seq.G_s, abs.(real.(S)), yaxis=:log)
#plot(seq.G_s, abs.(real.(S)))
#plot(seq.G_s, abs.(imag.(S)))


