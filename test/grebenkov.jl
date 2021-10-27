using DiffusionSimulator, CUDA, StatsBase, Plots, Gtk, JLD2

sfname = diff_save_interface("output/greb")

ra=10e-6; # [m]
N_ii=500; # number of side pixels
#len = collect(range(-ra*1.3, ra*1.3, length=N_ii))
len = collect(range(-20e-6, 20e-6, length=N_ii))
dgrid = zeros(N_ii, N_ii)
for i in 1:N_ii
    for j in 1:N_ii
        dgrid[i,j] = sqrt((len[i]^2) + len[j]^2)
    end
end
I = zeros(Int32, size(dgrid))
I[dgrid .>= ra] .= 1
phi = sum(I)/length(I)
gam=2.675e8

# Setup sequence
delta=5e-3
Delta=50e-3
tf=2*Delta
t=collect(range(0, tf, length=1000))
G= 0 .* t
dt=t[2]-t[1]
G[t .<= delta] .= 1
G[(t .>= Delta) .& (t .<= Delta+delta)] .= -1
dir = [1,1]./sqrt(2)
G = kron(G, dir')

N = round(100e6/phi)
seq = Seq(G, t, collect(0:0.001:0.58904)) # build seq object
simu = Simu([1e-9], N, abs(len[2]-len[1]), gam) # build sim object
println("dx = ", abs(len[2]-len[1]))

S = Array(diff_sim_gpu(I, seq, simu))


bval=((2.675e8.*seq.G_s .* delta).^2).*(Delta-delta/3);
bval .*= 1e-9
display(plot(bval, abs.(real.(S)), yaxis=:log))
res =Dict()
res["analytical"] = S



safe_save(sfname, res)




