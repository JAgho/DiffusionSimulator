using Plots, JLD2, Statistics

#a = JLD2.load("output/bases/basis_1e8.jld2")
#b = JLD2.load("output/bases/basis_1e9.jld2")
#c = JLD2.load("output/bases/basis_1000_1e8.jld2")

a = JLD2.load("output/bases/basis_200_dde_1.jld2")
b = JLD2.load("output/bases/basis_500_dde_1.jld2")
#c = JLD2.load("output/bases/basis_200_dde_2.jld2")
#d = JLD2.load("output/bases/basis_500_dde_2.jld2")

#mean(a["ord"])
Gₛ = collect(0:0.001:0.793)
for key in keys(a)
    am = mean(a[key])
    bm = mean(b[key])
    println(size(b[key]))
    #cm = mean(c[key])
    plt = plot(title = key, size=(1000,1000), xlabel="Max B field (T/m)", yaxis=:log, ylim=(1e-3, 1))
    #plot!(Gₛ, mean(hcat(a[key],c[key]),dims=2), label="1e8 particles, 200x200 basis extra")
    plot!(Gₛ, a[key])
    plot!(Gₛ, b[key])
    #plot!(Gₛ, c[key])
    #plot!(Gₛ, d[key])
    #plot!(Gₛ, mean(hcat(b[key],d[key]),dims=2), label="1e8 particles, 500x500 basis extra")
    #plot!(Gₛ, cm, label="1e8 particles, 1000x1000 basis")
    display(plt)
end



