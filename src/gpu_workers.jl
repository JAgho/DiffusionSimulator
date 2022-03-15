using Distributed
device_pool = [2,3,4,5,6,7]
addprocs(length(device_pool)-1)

@everywhere using CUDA
@everywhere using DiffusionSimulator

work = pushfirst!(workers(), 1)
asyncmap((zip(work, device_pool))) do (p, d)
    remotecall_wait(p) do
        @info "Worker $p uses $d"
        CUDA.device!(d)
    end
end