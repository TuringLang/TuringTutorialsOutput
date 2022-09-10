
@mini_model function m(x)
    a ~ Normal(0.5, 1)
    b ~ Normal(a, 2)
    x ~ Normal(b, 0.5)
    return nothing
end


# First, we import some needed packages
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


struct SamplingContext{R<:Random.AbstractRNG,S<:AbstractMCMC.AbstractSampler}
    rng::R
    sampler::S
end

struct DummySampler <: AbstractMCMC.AbstractSampler end

SamplingContext() = SamplingContext(Random.default_rng(), DummySampler())

function observe(context::SamplingContext, sampler, varinfo, dist, var_id, var_value)
    logp = Distributions.loglikelihood(dist, var_value)
    varinfo[var_id] = (var_value, logp)
    return var_value # return value not used, only for coherent behavior as `assume`
end

function assume(context::SamplingContext, sampler::DummySampler, varinfo, dist, var_id)
    sample = Random.rand(context.rng, dist)
    logp = Distributions.loglikelihood(dist, sample)
    varinfo[var_id] = (sample, logp)
    return sample
end;


macro mini_model(expr)
    return esc(mini_model(expr))
end

function mini_model(expr)
    def = MacroTools.splitdef(expr)

    # `MacroTools.postwalk` is a utility function for traverse AST in post-order
    def[:body] = MacroTools.postwalk(def[:body]) do sub_expr
        # `MacroTools.@capture` provide a utility function for pattern matching
        if MacroTools.@capture(sub_expr, var_ ~ dist_)
            if var in def[:args]
                return quote
                    $(observe)(
                        context,
                        context.sampler,
                        varinfo,
                        $dist,
                        $(Meta.quot(var)),
                        $var,
                    )
                end
            else
                return quote
                    $var = $(assume)(
                        context, context.sampler, varinfo, $dist, $(Meta.quot(var))
                    )
                end
            end
        else
            return sub_expr
        end
    end

    def[:args] = vcat([
        # Insert extra arguments
        :(varinfo),
        :(context),
    ], def[:args])

    return MacroTools.combinedef(def)
end;


struct MiniModel{F,D} <: AbstractMCMC.AbstractModel
    f::F
    data::D # a NamedTuple of all the data
end


# Just a Normal distribution as proposal in this simple case
struct MHSampler{P<:Normal} <: AbstractMCMC.AbstractSampler
    proposal::P
end

struct Transition{V}
    varinfo::V
end

SamplingContext(sampler::MHSampler) = SamplingContext(Random.default_rng(), sampler)


function assume(context::SamplingContext, sampler::MHSampler, varinfo, dist, var_id)
    old_value = varinfo.values[var_id]
    # propose a random-walk step, i.e, add the current value to a random 
    # value sampled from a Normal distribution centered at 0
    value = old_value + rand(context.rng, sampler.proposal)
    logp = Distributions.loglikelihood(dist, value)
    varinfo[var_id] = (value, logp)
    return value
end;


# The fist step
function AbstractMCMC.step(
    rng::Random.AbstractRNG, model::MiniModel, sampler::MHSampler; kwargs...
)
    vi = VarInfo()
    ctx = SamplingContext() # context with DUmmySampler
    model.f(vi, ctx, values(model.data)...)
    return Transition(vi), vi
end

# Defining functions to compute values to determine if accept the proposal
function condition_logprob(vi, condition_vi, args, proposal)
    return sum(
        Distributions.loglikelihood(proposal, vi.values[key] - condition_vi.values[key]) for
        key in keys(vi.values) if !(key in args)
    )
end

log_joint(vi) = sum(values(vi.logps))

function logα(old_vi, new_vi, args, proposal)
    return log_joint(new_vi) - log_joint(old_vi) +
           condition_logprob(old_vi, new_vi, args, proposal) -
           condition_logprob(new_vi, old_vi, args, proposal)
end

# The following steps
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::MiniModel,
    sampler::MHSampler,
    prev_state::VarInfo, # is just the old vi
    kargs...,
)
    vi = prev_state
    new_vi = deepcopy(vi)
    ctx = SamplingContext(sampler)
    model.f(new_vi, ctx, values(model.data)...)
    log_α = logα(vi, new_vi, keys(model.data), ctx.sampler.proposal)
    if -randexp(rng) < log_α
        return Transition(new_vi), new_vi
    else
        return Transition(prev_state), prev_state
    end
end

# Create Chains object at the end of the sampling process
function AbstractMCMC.bundle_samples(
    samples, model::MiniModel, ::MHSampler, ::Any, ::Type; kargs...
)
    # We get a vector of `Transition`s
    values = [sample.varinfo.values for sample in samples]
    params = [key for key in Base.keys(values[1]) if key ∉ keys(model.data)]
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


# Manually defining the model.
model = MiniModel(m, (x=3.0,))

# `sample` function is implemented as part of `AbstractMCMC.jl`
samples = sample(model, MHSampler(Normal(0, 1)), 1000000)


using Turing

@model function turing_m(x)
    a ~ Normal(0.5, 1)
    b ~ Normal(a, 2)
    x ~ Normal(b, 0.5)
    return nothing
end;


sample(turing_m(3.0), MH(), 1000000)

