---
redirect_from: "tutorials/14-minituring/"
title: "MiniTuring"
permalink: "/:collection/:name/"
---


In this session, we will develop a very simple probabilistic programming language.
The implementation is similar to that of [Turing.jl's](https://turing.ml/dev/).
This is intentional, as we want to demonstrate some key ideas from Turing's internal implementation.

To make things easy to understand and to implement.
We will restrict our language to a very simple subset of the language that Turing actually supports.
Defining an accurate syntax description is not our goal here, instead, we will give a simple example, and all program that "looks like" this should work.

For a model defined by the following mathematical form. (Consider `x` is data, or observed variable in this example.)

$$
\begin{align}
a &\sim Normal(0.5, 1) \\
b &\sim Normal(a, 2) \\
x &\sim Normal(b, 0.5)
\end{align}
$$

We will have the following model definition in our small language.

```julia
@mini_model function m(x)
    a ~ Normal(0.5, 1)
    b ~ Normal(a, 2)
    x ~ Normal(b, 0.5)
    return nothing
end
```



Specifically,

  - All observed variables should be arguments of the program.
  - No control flow is permitted in the model definition.
  - All variables should be scalars.
  - No function return.

```julia
# First, we import some needed packages
using MacroTools, Distributions, Random, AbstractMCMC, MCMCChains
```




Before getting to the actually compiler, let's first build the data structure for program trace.
A program trace for a probabilistic programming language need to at least record the values of stochastic variables and (log)probability of the corresponding stochastic variables scored by the prior distributions.

> **Julia Sidebar**
> Julia's high-performance is relied upon its multiple-dispatched based compilation scheme.
> In so many words, Julia compiler will produce a compiled version of the program customized for some specific types.
> Thus, in the idea case, it should have very similar performance as a pre-compiled C program.
> The implication of this is that although typing is not necessary for correct program execution, performance will heavily rely on programmer providing correct and coherent type information.

```julia
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
```




Now, let's define a `context` that does something very simple: if a stochastic variable is observed, we compute its log probability; or, if it is assumed(sampled), then we sample from prior and also computes its log probability.
`context` is an Turing internal implementation mechanism.
Because different inference algorithms may have slightly different actions to do with stochastic variables, we want a good abstraction and interface to implement these.
The name `context` comes from the "contextual dispatching" idea of [Cassette.jl](https://github.com/JuliaLabs/Cassette.jl).
Every `context`s is defined to be a type in Julia, and the `observe` and `assume` functions will take a `AbstractContext` type as their first arguments.
Then, at runtime, the Julia multiple-dispatch system will dispatch corresponding functions according to the type that is a specific `context`.

**Note:** *Although the context system in this tutorial is inspired by Turing's, Turing's context system is battle-tested and much more complicated.
Thus, readers are advised to refer to Turing's documentations and code if knowledge of Turing's context system is required.
Also, it should be pointed out that the context system design in Turing is not in its definitive form as of the current version at the time of this notebook is written.*

```julia
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
```




Then, we'll encounter our first nontrivial piece of Julia code, that is the compiler.
In one sentence, the compiler will replace every tilde notation with a call to `observe` or `assume` function defined in the `context` code block.

> **Julia Sidebar**
> I am afraid there is not easy way to explain what happens here. So I am going leave some references, and once the reader go though the referenced material, the following block should not be too difficult to understand.
> 
>   - [Julia Metaprogramming](https://docs.julialang.org/en/v1/manual/metaprogramming/)
>   - [MacroTools](https://fluxml.ai/MacroTools.jl/dev/pattern-matching/)
>   - [Turing Compiler Design](https://turing.ml/dev/docs/for-developers/compiler) (The implementation here is heavily simplified compared to Turing's compiler.)

```julia
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
```




Next, let's define a `MiniModel` struct.
In the actual implementation of the compiler in Turing, the model construction is done as the final step of the compilation.
But for the sake of simplicity, we will construct the model manually.
This also requires us to keep a field for the actually data in the `MiniModel` struct.

```julia
struct MiniModel{F,D} <: AbstractMCMC.AbstractModel
    f::F
    data::D # a NamedTuple of all the data
end
```




To illustrate how inference work for our mini language. We will implement an extremely simplistic Random-Walk Metropolis-Hastings sampler.

> **Julia Sidebar**
> We used the Julia package [AbstractMCMC.jl](https://github.com/TuringLang/AbstractMCMC.jl) here.
> For the interface design, we refer the reader to the [documentation](https://beta.turing.ml/AbstractMCMC.jl/dev/).

In simple words, we need to implement a sub-type of `AbstractModel`, which we did in the code block above; a sub-type of `AbstractSampler`, which we'll do in the next code block; and `step` functions, following the sampler definition.

```julia
# Just a Normal distribution as proposal in this simple case
struct MHSampler{P<:Normal} <: AbstractMCMC.AbstractSampler
    proposal::P
end

struct Transition{V}
    varinfo::V
end

SamplingContext(sampler::MHSampler) = SamplingContext(Random.default_rng(), sampler)
```

```
Main.##WeaveSandBox#305.SamplingContext
```





We hard-coded the proposal step as part of a `context` implementation.
In Turing, to accommodate more use-cases, the `proposal` function is abstracted out.

```julia
function assume(context::SamplingContext, sampler::MHSampler, varinfo, dist, var_id)
    old_value = varinfo.values[var_id]
    # propose a random-walk step, i.e, add the current value to a random 
    # value sampled from a Normal distribution centered at 0
    value = old_value + rand(context.rng, sampler.proposal)
    logp = Distributions.loglikelihood(dist, value)
    varinfo[var_id] = (value, logp)
    return value
end;
```




We need to define two `step` functions, one for the first step and the other for the following steps. The two functions are identified with different arguments they take.

```julia
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
```




Now, let's see how our mini probabilistic programming language works.
Define the probabilistic model.

```julia
@mini_model function m(x)
    a ~ Normal(0.5, 1)
    b ~ Normal(a, 2)
    x ~ Normal(b, 0.5)
    return nothing
end;
```




Given data `x = 3.0`.

```julia
# Manually defining the model.
model = MiniModel(m, (x=3.0,))

# `sample` function is implemented as part of `AbstractMCMC.jl`
samples = sample(model, MHSampler(Normal(0, 1)), 1000000)
```

```
Chains MCMC chain (1000000×2×1 Array{Float64, 3}):

Iterations        = 1:1:1000000
Number of chains  = 1
Samples per chain = 1000000
parameters        = a, b

Summary Statistics
  parameters      mean       std   naive_se      mcse           ess      rh
at
      Symbol   Float64   Float64    Float64   Float64       Float64   Float
64

           a    0.9768    0.8985     0.0009    0.0030    82324.1752    1.00
00
           b    2.8807    0.4881     0.0005    0.0011   174464.7328    1.00
00

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           a   -0.7865    0.3695    0.9778    1.5836    2.7340
           b    1.9272    2.5497    2.8800    3.2113    3.8348
```





Now, let's see what Turing gives us.

```julia
using Turing

@model function turing_m(x)
    a ~ Normal(0.5, 1)
    b ~ Normal(a, 2)
    x ~ Normal(b, 0.5)
    return nothing
end;
```


```julia
sample(turing_m(3.0), MH(), 1000000)
```

```
Chains MCMC chain (1000000×3×1 Array{Float64, 3}):

Iterations        = 1:1:1000000
Number of chains  = 1
Samples per chain = 1000000
Wall duration     = 11.28 seconds
Compute duration  = 11.28 seconds
parameters        = a, b
internals         = lp

Summary Statistics
  parameters      mean       std   naive_se      mcse          ess      rha
t   ⋯
      Symbol   Float64   Float64    Float64   Float64      Float64   Float6
4   ⋯

           a    0.9690    0.8934     0.0009    0.0042   45125.3067    1.000
0   ⋯
           b    2.8792    0.4881     0.0005    0.0019   66869.0960    1.000
0   ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           a   -0.7738    0.3680    0.9676    1.5710    2.7301
           b    1.9253    2.5471    2.8803    3.2117    3.8319
```





As you can see, we got the same results as Turing.
