 


function get_devices_memory()
    dev = CUDA.devices()
    res = [0 for i in 1:length(dev)]
    for i in 0:length(dev)-1
        CUDA.@sync CUDA.device!(i)
        
        res[i] = CUDA.total_memory() - CUDA.available_memory()
    end
    return res
end

println.(get_devices_memory() ./ 2^30, "Gb free")