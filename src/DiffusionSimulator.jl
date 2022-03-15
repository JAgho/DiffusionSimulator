module DiffusionSimulator
using StatsBase, CUDA, Adapt#, Plots
#import Gtk: save_dialog, Null, GtkFileFilter
import JLD2: save
export Seq, Simu, diff_sim, mask_pos, diff_sim_gpu#, diff_save_interface, safe_save
# Write your package code here.
struct Seq
    G::Array{Float32, 2}
    t::Array{Float32, 1}
    G_s::Array{Float64, 1}
end

struct Simu
    D::Array{Float32, 1}
    N::Int32
    r::Float32
    γ::Float64
end

include("util.jl")
#include("interface.jl")

function diff_sim(I,seq,simu)
    t=seq.t;
    G=seq.G;
    D=simu.D
    N=simu.N;
    r=simu.r;

    dt=t[2]-t[1]
    gam=2.675e8
    N_i=size(I)
    dim=length(N_i)
    dx=sqrt.(2*dim*dt.*D)
    N_c=sum(unique(I[I .> 0]))
    
    N=round(Int64, N*length(I)/sum(I .> 0))
    #println(N)
    y = (r .* N_i)
    X=(rand(N, dim)) .* [y[1] y[2]];
    ind, A = mask_pos!(X, N_i, r)

    kill = I[ind] .== 0
    #deleteat!(view(X,:, 1), kill); deleteat!(view(X,:, 2), kill)
    X = X[.!kill, :]
    deleteat!(ind, kill)
    #println(length(X))
    #println(length(ind))
    ind_s, A = mask_pos!(X, N_i, r)
    N_p = length(ind)
    pre_dx = zeros(N_p) .+ dx
    pre_ang = zeros(N_p)
    ind_p = similar(ind_s)
    B = similar(A)
    Xn = similar(X)
    Ie = similar(ind_s)
    phase = similar(pre_ang)
    phasesum = similar(X)
    Ind_n = similar(X)
    for tt in 1:length(t)
        #println("timestep = $tt")
        if dim==2
            pre_ang .= rand(N_p).*(2*pi)
            pol2cart!(pre_ang, pre_dx, Xn)
            ind_p, B=mask_pos(mod.(X.+Xn,N_i[1]*r),N_i,r, ind_s, B)
            Ie .= I[ind_s] .== I[ind_p];
            Xn[:,1] .*= Ie
            Xn[:,2] .*= Ie
            phasesum .= X .* G[tt,:]'
            #dump(phasesum)
            #println(size(sum(phasesum, dims=2))[:, 1])
            phase .= phase .+ vec(sum(phasesum, dims=2))
            M1=rem.(X,N_i[1]*r); 
            M2=mod.(X,N_i[1]*r)
            Ind_n .!= (M2.==X).*sign.(M1).*Ie
            phase .+= vec(sum(-Ind_n.*N_i[1]*r .* sum(G[1:tt,:]),dims=2))
            X.=M2
        end
        

    end

    # Need to sort out the complex exponentiation part of this
    #S=zeros(Complex, length(seq.G_s));
    #for gg=1:length(S)
    #    S[gg] .= mean(exp.((1im*gam*dt*seq.G_s[gg]).*phase))
    #end

    #Xn(:,1) = Xn(:,1).*Ie; Xn(:,2) = Xn(:,2).*Ie;
    #A, ind = mask_pos(X, N_i, r)
    return phase

end

function diff_sim_gpu(I,seq,simu)
    println("loading simulation...")
    ## Unpack variables and copy to device
    t=seq.t;
    G = CUDA.zeros(Float32, size(seq.G))
    G = seq.G
    D=simu.D
    N=simu.N;
    r=cu(Float32(simu.r))
    dt=t[2]-t[1]
    N_i = size(I)
    dim=length(N_i)
    dx=Float32.(sqrt.(2*dim*dt.*D))
    y = (r .* N_i)
    u = N_i[1]*r
    ## Compute intial positions on the host
    X_cpu=rand(Float32, N, dim) .* [y[1] y[2]]
    ind, A = mask_pos!(X_cpu, N_i, r)
    kill = I[ind] .!= 0 # indices of points in diffusing regions only
    Xc = X_cpu[kill, 1]
    Yc = X_cpu[kill, 2]
    N_p = length(Xc)
    println(N_p, " active particles")
    ## Allocate device memory
    X = CUDA.zeros(Float32, length(Xc))
    Y = CUDA.zeros(Float32, length(Yc))
    copyto!(X, Xc); copyto!(Y, Yc) # copy host coords to device
    ind_s, A, B = mask_pos_a((X, Y), N_i, r)
    pre_dx = CUDA.fill(dx[1], N_p)
    pre_ang = CUDA.zeros(Float32, N_p)
    Ie = CUDA.zeros(Bool, N_p)
    phase = CUDA.zeros(Float32, N_p)
    ind_p = similar(ind_s)
    X1 = similar(X) 
    X2 = similar(X) 
    Y1 = similar(X)
    Y2 = similar(X)
    Xn = similar(X) 
    Yn = similar(X) 
    Ind_nx = CUDA.zeros(Float32, N_p)
    Ind_ny = CUDA.zeros(Float32, N_p)
    C1 = CUDA.zeros(Bool, N_p)
    C2 = CUDA.zeros(Bool, N_p)
    I = cu(I)
    println("simulation variables loaded!")
    ## Begin main loop
    for tt in 1:length(t)
        ## Generate random moves
        CUDA.rand!(pre_ang)
        pre_ang .*= (2*pi)
        pol2cart!(pre_ang, pre_dx, Xn, Yn)
        ## Apply trial steps to coords
        X1 .= X; Y1 .= Y
        X1 .+= Xn; Y1 .+= Yn
        ## Compute the cell that trial steps end within
        cmod(X2, X1, u); cmod(Y2, Y1, u)
        ind_p, A, B = mask_pos_a((X2, Y2), N_i, r, ind_p, A[1], A[2], B[1], B[2])
        A[1] .= I[ind_s] # this is also where we should populate pre_dx?
        A[2] .= I[ind_p]
        ## Recognise illegal steps and delete these moves
        ## TODO write kernel to allow true comparison of cell moves
        ## This should allow probablistic acceptance on basis of permeability
        Ie .= (A[1] .== A[2])
        Xn .*= Ie;  Yn .*= Ie
        ## Finalise moves    
        X .+= Xn;   Y .+= Yn
        ## Compute phase change based on position
        X2 .= X .* G[tt,1]; Y2 .= Y .* G[tt,2]
        phase .+= X2; phase.+= Y2
        ## Apply periodic boundary conditions     
        crem(X1, X, u); crem(Y1, Y, u)
        cmod(X2, X1, u); cmod(Y2, Y1, u)
        A[1] .= sign.(X1); A[2] .= sign.(Y1)
        C1 .= X2 .!= X; C2 .= Y2 .!= Y 
        ## Modify phase change to account for PBC
        ax = sum(G[1:tt,1]) * u; ay = sum(G[1:tt,2]) * u 
        #Ind_nx .= A[1] .* C1 .* Ie; Ind_ny .= A[2] .* C2 .* Ie
        Ind_nx .= A[1] .* C1; Ind_ny .= A[2] .* C2
        Ind_nx .*= ax; Ind_ny .*= ay 
        phase .-= Ind_nx 
        phase .-= Ind_ny 
        ## Save this time-step's move
        X .= X2; Y .= Y2      
        # if mod(tt, 100) == 0
        #     println("timestep = $tt")
        #     #display(Plots.scatter(X[1:100], Y[1:100], xlims=[0,r*N_i[1]], ylims=[0,r*N_i[2]], ratio=:equal))
        # end
    end
    S = zeros(Float64, length(seq.G_s))
    
    for i in 1:length(S)
        S[i] = mean(cos.(Float64.(phase) .* simu.γ .* dt .* seq.G_s[i]))
    end
    return Array(S)
end

end