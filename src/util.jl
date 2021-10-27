
function pol2cart(th,r)
    x = r.*cos.(th)
    y = r.*sin.(th)
    return (x, y)
end

function pol2cart!(th,r, X)
    X = vec(X)
    n = length(X)
    e = n ÷ 2
    X[1:e] .= r.*cos.(th)
    X[e+1:end] .= r.*sin.(th)
end


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
        i =1
        @inbounds for (f, g) in zip(view(A,:,1), view(A,:,2))
           ind[i] = f + (g-1)*N_ii[1]
           i += 1
        end
    end
end

function mask_pos!(X::Array{Float32,2},  N_ii, r, ind=ones(Int64, max(size(X)...)), A=zeros(Int64, size(X)))
    A.=ceil.(X .* (1 ./ r))
    bound!(A, N_ii)
    sub2ind!(ind, A, N_ii)
    return ind, A
end

############################

# CUDA implementations of helper functions

############################

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
    lbound(A1, lb); lbound(A2, lb)
    ubound(A1, n1); ubound(A2, n2)
    A2 .-= 1
    A2 .*= n1
    ind .= A1 .+ A2
    return ind, (A1, A2), (B1, B2)
end

function pol2cart!(th,r, X::CuArray{Float32,1}, Y::CuArray{Float32,1})
    X .= cos.(th)
    X .*=  r
    Y .= sin.(th)
    Y .*= r
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