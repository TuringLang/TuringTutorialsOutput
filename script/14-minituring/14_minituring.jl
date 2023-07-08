
@mini_model function m(x)
    a ~ Normal(0.5, 1)
    b ~ Normal(a, 2)
    x ~ Normal(b, 0.5)
    return nothing
end


using MacroTools, Distributions, Random, AbstractMCMC, MCMCChains


struct VarInfo{V,L}
    values::V
    logps::L
end

VarInfo() = VarInfo(Dict{Symbol,Float64}(), Dict{Symbol,Float64}())

function Base.setindex!(varinfo::VarInfo, (value, logp), var_id)
    varinfo.values[var_id] = value
    varinfo.logps[var_id] = logp
    return varinfo
end


struct SamplingContext{S<:AbstractMCMC.AbstractSampler,R<:Random.AbstractRNG}
    rng::R
    sampler::S
end

struct PriorSampler <: AbstractMCMC.AbstractSampler end

function observe(context::SamplingContext, varinfo, dist, var_id, var_value)
    logp = logpdf(dist, var_value)
    varinfo[var_id] = (var_value, logp)
    return nothing
end

function assume(context::SamplingContext{PriorSampler}, varinfo, dist, var_id)
    sample = Random.rand(context.rng, dist)
    logp = logpdf(dist, sample)
    varinfo[var_id] = (sample, logp)
    return sample
end;


macro mini_model(expr)
    return esc(mini_model(expr))
end

function mini_model(expr)
    # Split the function definition into a dictionary with its name, arguments, body etc.
    def = MacroTools.splitdef(expr)

    # Replace tildes in the function body with calls to `assume` or `observe`
    def[:body] = MacroTools.postwalk(def[:body]) do sub_expr
        if MacroTools.@capture(sub_expr, var_ ~ dist_)
            if var in def[:args]
                # If the variable is an argument of the model function, it is observed
                return :($(observe)(context, varinfo, $dist, $(Meta.quot(var)), $var))
            else
                # Otherwise it is unobserved
                return :($var = $(assume)(context, varinfo, $dist, $(Meta.quot(var))))
            end
        else
            return sub_expr
        end
    end

    # Add `context` and `varinfo` arguments to the model function
    def[:args] = vcat(:varinfo, :context, def[:args])

    # Reassemble the function definition from its name, arguments, body etc.
    return MacroTools.combinedef(def)
end;


struct MiniModel{F,D} <: AbstractMCMC.AbstractModel
    f::F
    data::D # a NamedTuple of all the data
end


struct MHSampler{T<:Real} <: AbstractMCMC.AbstractSampler
    sigma::T
end

MHSampler() = MHSampler(1)

function assume(context::SamplingContext{<:MHSampler}, varinfo, dist, var_id)
    sampler = context.sampler
    old_value = varinfo.values[var_id]

    # propose a random-walk step, i.e, add the current value to a random 
    # value sampled from a Normal distribution centered at 0
    value = rand(context.rng, Normal(old_value, sampler.sigma))
    logp = Distributions.logpdf(dist, value)
    varinfo[var_id] = (value, logp)

    return value
end;


# The fist step: Sampling from the prior distributions
function AbstractMCMC.step(
    rng::Random.AbstractRNG, model::MiniModel, sampler::MHSampler; kwargs...
)
    vi = VarInfo()
    ctx = SamplingContext(rng, PriorSampler())
    model.f(vi, ctx, values(model.data)...)
    return vi, vi
end

# The following steps: Sampling with random-walk proposal
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::MiniModel,
    sampler::MHSampler,
    prev_state::VarInfo; # is just the old trace
    kwargs...,
)
    vi = prev_state
    new_vi = deepcopy(vi)
    ctx = SamplingContext(rng, sampler)
    model.f(new_vi, ctx, values(model.data)...)

    # Compute log acceptance probability
    # Since the proposal is symmetric the computation can be simplified
    logα = sum(values(new_vi.logps)) - sum(values(vi.logps))

    # Accept proposal with computed acceptance probability
    if -randexp(rng) < logα
        return new_vi, new_vi
    else
        return prev_state, prev_state
    end
end;


function AbstractMCMC.bundle_samples(
    samples, model::MiniModel, ::MHSampler, ::Any, ::Type{Chains}; kwargs...
)
    # We get a vector of traces
    values = [sample.values for sample in samples]
    params = [key for key in keys(values[1]) if key ∉ keys(model.data)]
    vals = reduce(hcat, [value[p] for value in values] for p in params)
    # Composing the `Chains` data-structure, of which analyzing infrastructure is provided
    chains = Chains(vals, params)
    return chains
end;


@mini_model function m(x)
    a ~ Normal(0.5, 1)
    b ~ Normal(a, 2)
    x ~ Normal(b, 0.5)
    return nothing
end;


sample(MiniModel(m, (x=3.0,)), MHSampler(), 1_000_000; chain_type=Chains)


using Turing
using PDMats

@model function turing_m(x)
    a ~ Normal(0.5, 1)
    b ~ Normal(a, 2)
    x ~ Normal(b, 0.5)
    return nothing
end

sample(turing_m(3.0), MH(ScalMat(2, 1.0)), 1_000_000)

