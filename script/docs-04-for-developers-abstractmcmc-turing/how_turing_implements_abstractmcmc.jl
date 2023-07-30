
using Turing

@model function gdemo(x, y)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    return y ~ Normal(m, sqrt(s²))
end

mod = gdemo(1.5, 2)
alg = IS()
n_samples = 1000

chn = sample(mod, alg, n_samples)


chn = sample(mod, alg, n_samples)


mod = gdemo(1.5, 2)


stepsize = 0.01
L = 10
alg = HMC(stepsize, L)


alg = IS()


function Sampler(alg::IS, model::Model, s::Selector)
    info = Dict{Symbol,Any}()
    state = ISState(model)
    return Sampler(alg, info, s, state)
end


mutable struct SamplerState{VIType<:VarInfo} <: AbstractSamplerState
    vi::VIType
end


mutable struct ISState{V<:VarInfo,F<:AbstractFloat} <: AbstractSamplerState
    vi::V
    final_logevidence::F
end

# additional constructor
ISState(model::Model) = ISState(VarInfo(model), 0.0)


struct Transition{T,F<:AbstractFloat}
    θ::T
    lp::F
end


function DynamicPPL.assume(rng, spl::Sampler{<:IS}, dist::Distribution, vn::VarName, vi)
    r = rand(rng, dist)
    push!(vi, vn, r, dist, spl)
    return r, 0
end


function DynamicPPL.observe(spl::Sampler{<:IS}, dist::Distribution, value, vi)
    return logpdf(dist, value)
end

