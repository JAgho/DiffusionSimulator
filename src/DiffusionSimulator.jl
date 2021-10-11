module DiffusionSimulator
using StatsBase, CUDA, Adapt, Plots
export Seq, Simu, diff_sim, mask_pos, diff_sim_gpu
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
end

include("util.jl")

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
    X_cpu=rand(Float32, N, dim) .* [y[1] y[2]]
    ind, A = mask_pos!(X_cpu, N_i, r)
    kill = I[ind] .!= 0
    #println("values at: ", I[ind[1:10]], " will be deleted according to mask: ", kill[1:10])
    #deleteat!(view(X,:, 1), kill); deleteat!(view(X,:, 2), kill)
    N_p = sum(kill)
    #dump(X[kill, 1])
    #println(size(view(X, :, 1)[kill]))
   
    Xc = X_cpu[kill, 1]
    Yc = X_cpu[kill, 2]
    #display(Plots.scatter(Xc[1:100], Yc[1:100], xlims=[0,r*N_i[1]], ylims=[0,r*N_i[2]], ratio=:equal))
    #println(size(X), size(X2))
    X = CUDA.zeros(Float32, length(Xc))
    Y = CUDA.zeros(Float32, length(Yc))
    copyto!(X, Xc); copyto!(Y, Yc)
    #X1[:,1] .= X[kill, 1]
    #X1[:,2] .= X[kill, 2]
    # deleteat!(ind, kill)
    # println(length(X))
    # println(length(ind))
    #N_i = CuArray(Float32.([size(I)[1], size(I)[2]]))
    ind_s, A, B = mask_pos_a((X, Y), N_i, r)
    #ind_test, D = mask_pos!(hcat(Xc, Yc), N_i, r)
    println(N_p, " active particles")
    
    pre_dx = CUDA.fill(dx[1], N_p)
    pre_ang = CUDA.zeros(Float32, N_p)
    ind_p = similar(ind_s)

    # Xw = similar(X)
    Ie = CUDA.zeros(Bool, N_p)
    phase = similar(pre_ang)
    phase .= 0
    phasesum = similar(X)
    X1 = similar(X) # working memory for coordinate manipulation
    X2 = similar(X) # this is tragically untidy, but fast
    Y1 = similar(X)
    Y2 = similar(X)
    Xn = similar(X) # memory for steps must be stable over simulation steps
    Yn = similar(X) # gpu programming really hates memory moves
    Ind_nx = CUDA.zeros(Float32, N_p)
    Ind_ny = CUDA.zeros(Float32, N_p)
    C1 = CUDA.zeros(Bool, N_p)
    C2 = CUDA.zeros(Bool, N_p)
    #display(Plots.scatter(X[1:100], Y[1:100], xlims=[0,r*N_i[1]], ylims=[0,r*N_i[2]], ratio=:equal))
    I = cu(I)
    println("simulation variables loaded!", typeof(I))
    u = N_i[1]*r
    for tt in 1:length(t)
         #println("timestep = $tt")
  
        #pre_ang .= rand(N_p).*(2*pi)
        CUDA.rand!(pre_ang)
        pre_ang .*= (2*pi)
        pol2cart!(pre_ang, pre_dx, Xn, Yn)
        X1 .= X; Y1 .= Y
        X1 .+= Xn; Y1 .+= Yn

        #map!(x->CUDA.mod(x, u), X2, X1)
        #map!(x->CUDA.mod(x, u), Y2, Y1)  #Map is explicitly asychronous... ew
        cmod(X2, X1, u); cmod(Y2, Y1, u)
        CUDA.@sync ind_p, A, B = mask_pos_a((X2, Y2), N_i, r, ind_p, A[1], A[2], B[1], B[2])

        A[1] .= I[ind_s]
        A[2] .= I[ind_p]
        Ie .= (A[1] .== A[2])

        Xn .*= Ie;  Yn .*= Ie
        
        X .+= Xn;   Y .+= Yn

        X2 .= X .* G[tt,1]; Y2 .= Y .* G[tt,2]

        phase .+= X2; phase.+= Y2
        
        crem(X1, X, u); crem(Y1, Y, u)
        cmod(X2, X1, u); cmod(Y2, Y1, u)
        #CUDA.@sync map!(x->CUDA.rem(x, u), X1, X)
        #CUDA.@sync map!(x->CUDA.rem(x, u), Y1, Y)
        #CUDA.@sync map!(x->CUDA.mod(x, u), X2, X)
        #CUDA.@sync map!(x->CUDA.mod(x, u), Y2, Y)
        # map!(x->CUDA.rem(x, u), X1, X)
        # map!(x->CUDA.rem(x, u), Y1, Y)
        # map!(x->CUDA.mod(x, u), X2, X)
        # map!(x->CUDA.mod(x, u), Y2, Y)

        A[1] .= sign.(X1); A[2] .= sign.(Y1)

        C1 .= X2 .!= X; C2 .= Y2 .!= Y 

        Ind_nx .= A[1] .* C1 .* Ie; Ind_ny .= A[2] .* C2 .* Ie

        ax = sum(G[1:tt,1]) * u; ay = sum(G[1:tt,2]) * u 

        Ind_nx .*= ax; Ind_ny .*= ay #turned out it needed to be 2d

        #println(A[1][1], "\t", X[1], "\t",  X1[1], "\t", X2[1], "\t", Ind_nx[1], "\t", sum(Ie))

        phase .-= Ind_nx
        phase .-= Ind_ny

        X .= X2; Y .= Y2
        
        if mod(tt, 100) == 0
            println("timestep = $tt")
            #display(Plots.scatter(X[1:100], Y[1:100], xlims=[0,r*N_i[1]], ylims=[0,r*N_i[2]], ratio=:equal))
            #o = sum(Ie)
            #println(o, " active particles")
            #println(Xn[1:10])
        end
    end
    return phase
end

end

"""

function mask_pos(X,N_i,r)
    X=ceil.(X .* 1 ./ r);
    Threads.@threads @simd for e in X
        if e < 1
            e = 1
        end
    end
    @code_lowered tmap(X)

    X[X.<1].=1;
    X[X[:,1]>N_i[1],1]=N_i(1);
    X[X[:,2]>N_i[2],2]=N_i(2);
    
    if length(N_i)==2
        %ind_s=sub2ind(N_i,X(:,1),X(:,2));
        ind_s = X(:,1) + (X(:,2)-1)*N_i(1);
    elseif length(N_i)==3
        X(X(:,3)>N_i(3),3)=N_i(3);
        ind_s=sub2ind(N_i,X(:,1),X(:,2),X(:,3));
    end
    
end
"""



