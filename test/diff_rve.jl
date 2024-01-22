using DiffusionSimulator, CUDA, Images, Plots
include("../src/sequences.jl")
#println("devices are: ", CUDA.devices())
#println("trying to print devices")
#sleep(10)
#CUDA.device!(1)
function main()
    #ldir = "../YTRecon/output/recon_error_dde/"
    
    #labels = ["1e-3", "5e-1", "5e-2", "5e-3"]
    
    sdir = "output/demo/"
    froot = "../YTRecon/bases/rve_detailed/"
    # rs = [4,8,16]
    # ls = [250]
    # xs = [600, 1200, 1800]
    rs = [16]
    #ls = [100,200,400,600,800,1000,1200,1600,2000]
    ls = [102, 300,500,700,900]
    xs = [1200]
    #fnames = [froot*"_r$r"*"_l$l"*".png" for r in rs, l in ls]

    #ldir = "../YTRecon/output/recon_error_dde/disord_5e-1/"
    #fn = "disord_5e-1.jld2"
    for r in rs
        for l in ls
            for x in xs
                #load image
                lname = froot*"rve_r$r"*"_l$l"*".png"
                dx = (x/1000)*1e-6
                I = Gray.(load(lname)) .< 0.5

                #compose sequence


                #build simulation
                #dx = (2/3)*1e-6
                gam=2.675e8
                phi = sum(I)/length(I)
                N = round(1e6/phi)

                # build seq object
                simu = Simu([1e-9], N, dx, gam) # build sim object
                #seq = pgse(10e-3, 10.1e-3, 0:0.001:0.793, [0,1], 1000)
                δ = 5e-3
                Δ = 10e-3
                tₘ = 5.1e-3
                dir = [0,1]
                Gₛ = collect(0:0.001:0.793)
                seq = dde(δ, Δ, tₘ, 1000, Gₛ, dir)

                #run simulation
                S = Array(diff_sim_gpu(I, seq, simu))

                #store result
                sname = sdir*"rve_r$r"*"_l$l"*"_x$x"*".jld2"
                println("saving $sname")
                results = Dict("I"=>I, "result"=>S)
                return results
                #JLD2.save(sname, results)

            end
        end
    end
    
end

begin
    result = main()
    plt = plot(seq.t, seq.G, label=["Gx" "Gy"])
    im = heatmap(result["I"], aspect_ratio =:equal)
    signal = plot((result["result"]), yscale=:log, xlabel="Gradient Field (mT/m)", ylabel="Signal Intensity", legend=nothing)


    display(plot!(plt, im, signal))
end