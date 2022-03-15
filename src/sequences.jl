
function greb(delta, Delta, grange, dir, len)

    delta=10e-3
    Delta=10.1e-3
    dir = [0,1]
    len = 1000
    G = pgse(delta, Delta, len)
    return Seq(kron(G, dir'), t, collect(grange))
end

function pgse(delta, Delta, grange, dir, len)
    tf=Delta+delta
    t=collect(range(0, tf, length=len))
    G=0 .* t
    G[t .<= delta] .= 1
    G[(t .>= Delta) .& (t .<= Delta+delta)] .= -1
    #G = pgse(delta, Delta, len)
    return Seq(kron(G, dir'), t, collect(grange))
end

function dde(δ₁, Δ₁, δ₂, Δ₂, tₘ, len, grange, dir)
    tf = Δ₁ + Δ₂ + tₘ + δ₂
    t = collect(range(0, tf, length=len))
    G=0 .* t
    G[(t .>= 0) .& (t .<= δ₁)] .= -1
    G[(t .>= Δ₁) .& (t .<= Δ₁+δ₁)] .= 1
    G[(t .>= Δ₁+tₘ) .& (t .<= Δ₁+tₘ+δ₂)] .= 1
    G[(t .>= Δ₁+tₘ+Δ₂) .& (t .<= Δ₁+tₘ+Δ₂+δ₂)] .= -1
    G = kron(G, dir')
    Seq(G, t, grange)
end

function dde(δ, Δ, tₘ, len, grange, dir)
    return dde(δ, Δ, δ, Δ, tₘ, len, grange, dir)
end


#typical DDE 1k steps
δ = 5e-3
Δ = 10e-3
tₘ = 5.01e-3
dir = [0,1]
Gₛ = collect(0:0.001:0.793)
seq = dde(δ, Δ, tₘ, 1000, Gₛ, dir)

#typical PGSE 1k steps
pgse(10e-3, 10.1e-3, 0:0.001:0.793, [0,1], 1000)