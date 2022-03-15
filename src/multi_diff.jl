using Distributed
device_pool = [2,3,4,5,6,7]
addprocs(length(device_pool))

@everywhere using CUDA
@everywhere using DiffusionSimulator



function get_devices_memory()
    dev = CUDA.devices()
    res = [0 for i in 1:length(dev)]
    for i in 0:length(dev)-1
        CUDA.@sync CUDA.device!(i)
        
        res[i+1] = CUDA.available_memory()
    end
    return res
end

function get_free_devices(reserved)
    a = get_devices_memory()
    dev = CUDA.devices()
    devs = [i for i in 0:length(dev)-1]
    return devs[a.>reserved]
end





# assign devices to respective workers
# each worker is a seperate process running on a different logical core
asyncmap((zip(workers(), device_pool))) do (p, d)
    remotecall_wait(p) do
        @info "Worker $p uses $d"
        CUDA.device!(d)
    end
end

@everywhere function test_gpu(I, simu)
    I = cu(I)
    A = CUDA.zeros(Float32, 100,100)
    A .+= I
    A .+= simu.N
    println("operating on chunk!")
    sleep(10)
    println("finished!")
    return Array(A)
end

function main()
@everywhere I = [rand(100,100) for i in 1:10]

@everywhere begin
    include("../src/sequences.jl")
    I = [Int64.(rand(Bool, 100,100)) for i in 1:10]
    dx = (2/3)*1e-6
    gam=2.675e8
    phi = sum(I[1])/length(I[1])
    N = round(Int64, 1e6/phi)
    # build seq object

    simu = Simu([1e-9], N, dx, gam)# build sim object
    seq = pgse(10e-3, 10.1e-3, 0:0.001:0.793, [0,1], 1000)
end

# f(x,y) = x+y
# @everywhere curry(f, y) = x->f(x, y)
# @everywhere addxy(x, y) = x + y    
# @everywhere addxyz(x, y, z) = x + y + z
# pmap(curry(curry(addxyz, 5), 5), 1:5)
#pmap(x -> f(x, 5, 8), x)
#pmap(x->diff_sim_gpu(x, seq, simu), I)
#return @spawnat 2 test_gpu(I[1], simu)#diff_sim_gpu(I[1], seq, simu)
#CUDA.device!(2)
return @spawnat 2 diff_sim_gpu(I[1], seq, simu)
end
q = main()

# @everywhere function pgse_diff(I)
#     dx = (2/3)*1e-6
#     gam=2.675e8
#     phi = sum(I)/length(I)
#     N = round(10e6/phi)
#     # build seq object

#     simu = Simu([1e-9], N, dx, gam) # build sim object
#     seq = pgse(10e-3, 10.1e-3, 0:0.001:0.793, [0,1], 1000)
#     S = Array(diff_sim_gpu(I, seq, simu))
# end
# device_pool = [2,3,4,5,6,7]
# #eval(:(a = 2))
# #@spawnat 2 eval(:(q = pgse_diff(I[2])))
# @everywhere v = rand(Bool, 100,100)
# q = @spawnat 2 test_gpu(v)
isready(q)
fetch(q)



#pmap(test_gpu, I)