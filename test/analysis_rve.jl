using JLD2, Plots, YTRecon, StatsBase

begin
    sdir = "output/rve_detailed_big/"
    froot = "../YTRecon/bases/rve_detailed/"
    # rs = [4,8,16]
    # ls = [250, 500, 1000]
    # xs = [600, 1200]
    rs = [16]
    ls = [100,102,200,300,400,500,600,700,800,900,1000,1200,1600,2000]
    lsa = [100,200,400,600,800,1000,1200,1600,2000]
    lsb = [102, 300, 500, 700, 900]
    xs = [1200]

    function load_entry(dir, r, l, x)
        lname = dir*"rve_r$r"*"_l$l"*"_x$x"*".jld2"
        return JLD2.load(lname, "result")
    end

    function load_entry(dir, r, l, x, stem)
        lname = joinpath(dir, "$(stem)_r$(r)_l$(l)_x$(x).jld2")
        return JLD2.load(lname, "result")
    end

    function load_image(dir, r, l, x, stem)
        lname = joinpath(dir, "$(stem)_r$(r)_l$(l)_x$(x).jld2")
        return JLD2.load(lname, "I")
    end

    function radial_average(I)
        dims = size(I)
        mid = Tuple(round.(Int64, ([dims...].+1) ./ 2))
        shape = CartesianIndices(dims)
        rvals = shape .- CartesianIndex(mid)
        rs = zeros(size(I))
        norm(x,y) = sqrt(x*x + y*y)
        for ((i,j), idx) in zip(Tuple.(rvals), shape)
            rs[idx] = norm(i,j)
            
        end
        r_max = maximum(rs)
        bins = collect(0:1.0:ceil(r_max))
        histo = zeros(Int64, length(bins))
        av = zeros(Float64, length(bins))
        nbins = length(bins)
        rb = (nbins)/r_max 
        for idx in shape
            rIndex=floor(Int, rs[idx]*rb) +1
            if rIndex < nbins+1
                (histo[rIndex] += 1)
                (av[rIndex] += I[idx])
            end
            #bins[fld(Int64, rs[idx], 1.0)] += 1
        end
        av ./=  histo
    end


end


w = (100,100)
descs = ["SQ"]
p = (window_sq=w,)

gs = collect(0:0.001:0.793)

rg = rs[1] #fixed r
xg = xs[1] #fixed dx
stem = "rve"
conds = ((sdir, rg, ls[i], xg, stem) for i in 1:length(ls))
#res2 = [init_basis(load_image(cond...), descs, p) for cond in conds];
#JLD2.save("output/rve_detailed/ffts_final2.jld2", Dict("results"=>res2))
res = JLD2.load("output/rve_detailed/ffts_final2.jld2", "results");
res = [res[i][3]["SQ"].swindow for i in 1:length(res)]
begin
p2s = []
plt1 = plot( xaxis="Max Field (T/m)", title = "PGSE diffusion signal error vs basis", size = (1500,1000))
plt2 = plot(title="S2", xaxis="Distance (pixels)")

difb = load_entry(collect(conds)[end]...)
Ib = res[end]
rab = radial_average(abs.(Ib))./2
s2b = Ib./2
for (i,cond) in zip(collect(3:length(conds)-1), collect(conds)[3:end-1])

    r = cond[2]; l = cond[3]; x = cond[4]
    name = "r=$r; dims=$l x $l; dx = $x nm"
    I = res[i]
    println(size(I), "  ",i, "    ", name)
    plot!(plt1, gs, abs.(load_entry(cond...).-difb), label= name)
    plot!(plt2, abs.((radial_average(abs.(I))./2).-rab), title="S2", xaxis="Distance (pixels)", label = name)
    push!(p2s, heatmap((I./2).-s2b, aspect_ratio=:equal, colorbar=false, axis=nothing, border=:none, title=name, titlefontsize=10))
    
end
plt3 = plot(p2s...);
megaplot = plot(plt1, plt2, plt3, size = (1600, 1000));
display(megaplot)
end


begin
vars = []
w = []
Is = []
for (i,cond) in zip(collect(1:length(conds)), collect(conds)[1:end])
    push!(vars, load_entry(cond...))
    push!(w, cond[3]^2)
    push!(Is, res[i])
end
w = Float64.(w)
v = zeros(length(vars[1]), length(vars))
for i in 1:size(v, 2)
    v[:,i] .= vars[i]
end

conv = mean(v, weights(w), dims = 2)
avs = sum(abs2, v .- conv, dims = 1)[1,:]
end

plt4 = plot(ls[1:end], avs[1:end], yscale=:log, yaxis = "Squared Error", xaxis = "Pixel Count", title = "Diffusion Error", xlim=(0,2000))
display(plt4)

begin
siz = (length(vars)-1, size(Is[2], 1), size(Is[2], 2))
It = zeros(siz)
for i in 2:length(conds)
    It[i-1, :,:] .= Is[i]
end

avsq= radial_average(mean(It, weights(w[2:end]), dims = 1)[1,:,:])./2
sqmap = (map(radial_average, Is[2:end])./2)
siz2 = (length(sqmap), length(avsq))
sqs = zeros(siz2)
for i in 1:length(conds)-1
    sqs[i,:] .= sqmap[i] 
end

finals = zeros(length(conds)-1)
for i in 1:length(conds)-1
    finals[i] =  sum(abs2, sqs[i,:] .- avsq) 
end
end
plot(ls[2:end], finals[1:end], yscale=:log, yaxis = "Squared Error", xaxis = "Pixel Count", title = "S2 Error")


