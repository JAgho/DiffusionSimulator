
function pol2cart(th,r)
    x = r.*cos.(th)
    y = r.*sin.(th)
    return (x, y)
end

function pol2cart!(th,r, X)
    #println(length(r), "\t" ,length(x))

    X = vec(X)
    n = length(X)
    e = n ÷ 2
    #println("length of X = ", length(X), "\t length of r = ", length(r), "\t length of th = ", length(th))
    #println("size of X is ", size(X), "size of r is ", size(r), "size of th is ", size(th))
    X[1:e] .= r.*cos.(th)
    X[e+1:end] .= r.*sin.(th)
end


# function bound2d!(A, N_ii)
#     @inbounds for i in 1:2:length(A)
#         if A[i] < 1
#             A[i] = 1
#         elseif A[i] > N_ii[1]
#             A[i] = N_ii[1]
#         end
#         if A[i+1] < 1
#             A[i+1] = 1
#         elseif A[i+1] > N_ii[2]
#             A[i+1] = N_ii[2]
#         end     
#     end
# end

#function sub2ind2d!(ind, A, N_ii)
    #     if length(N_ii)==2
    #         #ind .= A[1,:] .+ (A[2,:].-1).*N_ii[1];
    #         t = 1
    #         @inbounds for i in 1:2:length(A)
    #            ind[t] = A[i] + (A[i+1]-1)*N_ii[1]
    #            t += 1
    #         end
    #     end
    # end
    
    # function mask_pos2d(X,  N_ii, r, A=zeros(Int64, size(X)), ind=zeros(Int64, max(size(X)...)))
    #     A.=ceil.(X .* (1 ./ r))
    #     bound2d!(A, N_ii)
    #     sub2ind2d!(ind, A, N_ii)
    #     #println(size(ind))
    
    #     return A, ind
    # end

function bound!(A, N_ii)
    @inbounds @simd for i in 1:length(A) ÷ 2
        if A[i] < 1
            A[i] = 1
        elseif A[i] > N_ii[1]
            A[i] = N_ii[1]
        end
    end
    @inbounds @simd for i in length(A) ÷ 2:length(A)
        if A[i] < 1
            A[i] = 1
        elseif A[i] > N_ii[2]
            A[i] = N_ii[2]
        end     
    end
end

function sub2ind!(ind, A, N_ii)
    if length(N_ii)==2
        #ind .= A[1,:] .+ (A[2,:].-1).*N_ii[1];
        i =1
        @inbounds for (f, g) in zip(view(A,:,1), view(A,:,2))
           ind[i] = f + (g-1)*N_ii[1]
           i += 1
        end
    end
end

# 

function mask_pos(X::Array{Float64,2},  N_ii, r, ind=zeros(Int64, max(size(X)...)), A=zeros(Int64, size(X)), B=zeros(Bool, size(X)))
    A.=ceil.(X .* (1 ./ r))
    B .= A .< 1
    A[B] .= 1

    view(B, :, 1) .= view(A,:,1).>N_ii[1]
    view(A, :, 1) .= N_ii[1]
    view(B, :, 2) .= view(A,:,2).>N_ii[2]
    view(A, :, 2) .= N_ii[2]
    ind .= view(A,:,1) .+ (view(A,:,2).-1).*N_ii[1]
    return ind, A, B
end

function mask_pos!(X::Array{Float32,2},  N_ii, r, ind=ones(Int64, max(size(X)...)), A=zeros(Int64, size(X)))
    A.=ceil.(X .* (1 ./ r))
    bound!(A, N_ii)
    sub2ind!(ind, A, N_ii)
    #X(X<1)=1;
    #X(X(:,1)>N_i(1),1)=N_i(1);
    #X(X(:,2)>N_i(2),2)=N_i(2);
    return ind, A
end



############################

# CUDA implementations of helper functions

############################

function bound!(A::CuArray{Float32,2}, N_ii)
    for i in 1:length(A) ÷ 2
        if A[i] < 1
            A[i] = 1
        elseif A[i] > N_ii[1]
            A[i] = N_ii[1]
        end
    end
    for i in length(A) ÷ 2:length(A)
        if A[i] < 1
            A[i] = 1
        elseif A[i] > N_ii[2]
            A[i] = N_ii[2]
        end     
    end
end

function sub2ind!(ind, A::CuArray{Float32,2}, N_ii)
    if length(N_ii)==2
        #ind .= A[1,:] .+ (A[2,:].-1).*N_ii[1];
        i =1
        @inbounds for (f, g) in zip(view(A,:,1), view(A,:,2))
           ind[i] = f + (g-1)*N_ii[1]
           i += 1
        end
    end
end

function mask_pos!(X::CuArray{Float32,2},  N_ii, r, ind=CUDA.ones(Int32, max(size(X)...)), A=CUDA.zeros(Int32, size(X)))
    A.=ceil.(X .* (1 ./ r))
    println("bounding")
    bound!(A, N_ii)
    sub2ind!(ind, A, N_ii)
    println("indexing")
    #X(X<1)=1;
    #X(X(:,1)>N_i(1),1)=N_i(1);
    #X(X(:,2)>N_i(2),2)=N_i(2);
    return ind, A
end

function mask_pos((X, Y),  N_ii, r, ind=CUDA.zeros(Int32, max(size(X)...)), A=CUDA.zeros(Int32, size(X)), B=CUDA.zeros(Bool, size(X)), C=CUDA.zeros(Float32, size(X)))

    
    
    C .= X
    C .*= (1 / r)
    A .= ceil.(C)
    #A.=ceil.(X .* (1 ./ r))
    B .= A .< 1
    #A[B] .= 1
    #B = reshape(B, :)
    #A = reshape(A, :)
    n = length(A)
    e = n ÷ 2
    xdim = N_ii[1]
    ydim = N_ii[2]
    B[1:e] = A[1:e].>xdim
    B[e+1:end] = A[e+1:e*2].>ydim
    A[1:e][B[1:e]] = xdim
    A[e+1:end][B[e+1:end]] = ydim

    view(A, :, 1) .= N_ii[1]
    
    view(A, :, 2) .= N_ii[2]
    ind = A[1:e] .+ (A[e+1:end].-1).*N_ii[1]
    A = reshape(A, :, 2)
    B = reshape(B, :, 2)
    return ind, A, B, C
end

function mask_pos_a(pos::Tuple{CuArray{Float32,1},CuArray{Float32,1}},  N_ii, r)
    n = length(pos[1])
    ind=CUDA.zeros(Int32, n) 
    A1=CUDA.zeros(Int32, n); A2 = CUDA.zeros(Int32, n)
    B1=CUDA.zeros(Float32, n); B2 = CUDA.zeros(Float32, n)
    B1 .= pos[1] .* (1/r)
    B2 .= pos[2] .* (1/r)
    B1 .= ceil.(B1) 
    B2 .= ceil.(B2)
    A1 .= Int32.(B1)
    A2 .= Int32.(B2)
    
    n1 = cu(Int32(N_ii[1]))
    n2 = cu(Int32(N_ii[2]))
    lb = Int32(1)
    #CUDA.map(x -> x < lb ? lb : x, A2); map(x -> x < lb ? lb : x, A2)
    #map(x -> x > n1 ? n1 : x, A1); map(x -> x > n2 ? n2 : x, A2)
    lbound(A1, lb); lbound(A2, lb)
    ubound(A1, n1); ubound(A2, n2)
    A2 .-= 1
    A2 .*= n1
    ind .= A1 .+ A2
    return ind, (A1, A2), (B1, B2)
end

function mask_pos_a(pos::Tuple{CuArray{Float32,1},CuArray{Float32,1}},  N_ii, r, ind, A1, A2, B1,B2)
    B1 .= pos[1] .* (1/r)
    B2 .= pos[2] .* (1/r)
    B1 .= ceil.(B1) 
    B2 .= ceil.(B2)
    A1 .= Int32.(B1)
    A2 .= Int32.(B2)
    n1 = Int32(N_ii[1])
    n2 = Int32(N_ii[2])
    lb = Int32(1)

    #CUDA.map!(x -> x < lb ? lb : x, A1, A1); CUDA.map!(x -> x < lb ? lb : x, A2, A2)
    #CUDA.map!(x -> x > n1 ? n1 : x, A1, A1); CUDA.map!(x -> x > n2 ? n2 : x, A2, A2)
    lbound(A1, lb); lbound(A2, lb)
    ubound(A1, n1); ubound(A2, n2)
    #BOUNDED
    A2 .-= 1
    A2 .*= n1
    ind .= A1 .+ A2

    return ind, (A1, A2), (B1, B2)
end

function pol2cart!(th,r, X::CuArray{Float32,1}, Y::CuArray{Float32,1})
    #println(length(r), "\t" ,length(x))
    #X = reshape(X, :)
    #print(size(X))
    #println("length of X = ", size(X), "\t length of r = ", size(r), "\t length of th = ", size(th))
    
    #print(" ")
    #println(size(r.*cos.(th)))
    #X[:,1] = view(X, :, 1)
    #view(X, :, 1) = r.*cos.(th)
    #view(X, :, 2) = r.*sin.(th)
    #X = vec(X)

    #
    X .= cos.(th)
    X .*=  r
    Y .= sin.(th)
    Y .*= r
    #println("size of X is ", size(X), "size of r is ", size(r), "size of th is ", size(th))
    #X[1:e] .= w
    #X[e+1:end] = r.*sin.(th)
end

function cmod(dst, src, u)
    map!(x->CUDA.mod(x, u), dst, src)
end

function crem(dst, src, u)
    map!(x->rem(x, u), dst, src)
end

function lbound(arr, lb)
    CUDA.map!(x -> x < lb ? lb : x, arr, arr)
end

function ubound(arr, ub)
    CUDA.map!(x -> x > ub ? ub : x, arr, arr)
end