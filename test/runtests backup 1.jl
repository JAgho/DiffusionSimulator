using DiffusionSimulator, CUDA, StatsBase, Plots

    ra=10e-6; # [m]
    N_ii=60; # number of side pixels
    #[X,Y]=meshgrid(linspace(-ra*1.3,ra*1.3,N_ii))
    len = collect(range(-ra*1.3, ra*1.3, length=N_ii))
    dgrid = zeros(N_ii, N_ii)
    for i in 1:N_ii
        for j in 1:N_ii
            dgrid[i,j] = sqrt((len[i]^2) + len[j]^2)
        end
    end
    I = zeros(Int32, size(dgrid))
    #d=sqrt(X.^2+Y.^2)
    #I=d<=ra
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
    #println("running...")
    
    G = kron(G, dir')
    sum(G[1:200,:])
    #display(plot(G))
    #println("run")
    N = round(1e6/phi)
    seq = Seq(G, t, collect(0:0.01:.5))
    simu = Simu([2.3e-9], N, abs(len[2]-len[1]))
# eps(Float64)
#     rem.(cu([-0.01,-0.01]), cu([0.1, 0.1]))
#   mod.(cu([-0.01,-0.01]), cu([0.1, 0.1]))
# w = cu([-0.01,-0.01])
 #q = cu([0.1, 0.1])
 #CUDA.rem.(w, q)
 #CUDA.mod.(w, q)
#     rem(-0.01, 0.1)
#     mod(-0.01, 0.1)
# CUDA.rem.(-0.01, 0.1)

# CUDA.mod(-0.01, 0.1)
# map(x->CUDA.mod(x, 0.1), w)
    #ccall("extern __nv_fmodf", llvmcall, Cfloat, (Cfloat, Cfloat), -0.01, -0.01)
    #display(plot(G))
    # G(t<=delta)=1;
    # G(t>=Delta&t<=Delta+delta)=-1;
    # dir=[0,1];
    # G=kron(dir,G');
    # seq.G=G; # gradient shape
    # seq.t=t; # gradient times
    # seq.G_s=0:0.01:.5; # gradient strength(s)

    # # Setup numerical stuff
    # simu.r=abs(len[2]-len[1]) # pixel resolution
    # simu.N=(1e6*u); # number of initial walkers
    # simu.D=2.3e-9; # diffusivity


    # # Run simulation
    #@time phase = diff_sim(I,seq,simu)

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
##S = map(x->x <= 0.0 ? typemin(S) : x, real(S))
plot(seq.G_s, abs.(real.(S)), yaxis=:log)
#plot(seq.G_s, abs.(real.(S)))
#plot(seq.G_s, abs.(imag.(S)))

#comp.i = phasec
#Comple
    #scatter(X[1:100], Y[1:100])
    #p = zeros(length(phase))
    #copyto!(p, phase)
    #ph = p .* (0 - 1im)
    #ph .*=  gam*dt

    #plot(real.(s))sum(u)
    #sum(u)
    #length(u)
    #mean(exp.(1im*phase))

    #@benchmark DiffusionSimulator.mask_pos!(s...)
     #print(s)
