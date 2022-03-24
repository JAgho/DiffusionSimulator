using DiffusionSimulator, CUDA, JLD2, Images
include("../src/sequences.jl")
#println("devices are: ", CUDA.devices())
#println("trying to print devices")
#sleep(10)
CUDA.device!(2)
function main()

    rng =1:10
    res = []
    Is = []
    ldir = "../YTRecon/output/recon_error/disord_1e0/"
    
    sdir = "output/recon_error_dde/"
    fn = "disord_1e0_sq.jld2"
    for i in rng
        #load image
        ln = "recon_sq_$i.png"
        lname = ldir*ln
        push!(Is, Gray.(load(lname)) .< 0.5)

        #compose sequence


        #build simulation
        dx = (2/3)*1e-6
        gam=2.675e8
        phi = sum(Is[i])/length(Is[i])
        N = round(100e6/phi)
         # build seq object

        simu = Simu([1e-9], N, dx, gam) # build sim object
        # seq = pgse(10e-3, 10.1e-3, 0:0.001:0.793, [0,1], 1000)
        δ = 5e-3
        Δ = 10e-3
        tₘ = 5.1e-3
        dir = [0,1]
        Gₛ = collect(0:0.001:0.793)
        seq = dde(δ, Δ, tₘ, 1000, Gₛ, dir)
        #run simulation
        S = Array(diff_sim_gpu(Is[i], seq, simu))

        #store result
        push!(res, S)
        println("saving simulation $i of 10")
    end
    sname = sdir*fn
    results = Dict("I"=>Is, "result"=>res)
    JLD2.save(sname, results)
    
end


main()