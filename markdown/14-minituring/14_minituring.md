---
redirect_from: "tutorials/14-minituring/"
title: "MiniTuring"
permalink: "/:collection/:name/"
---


In this tutorial we develop a very simple probabilistic programming language.
The implementation is similar to [DynamicPPL](https://github.com/TuringLang/DynamicPPL.jl).
This is intentional as we want to demonstrate some key ideas from Turing's internal implementation.

To make things easy to understand and to implement we restrict our language to a very simple subset of the language that Turing actually supports.
Defining an accurate syntax description is not our goal here, instead, we give a simple example and all similar programs should work.

Consider a probabilistic model defined by

$$
\begin{aligned}
a &\sim \operatorname{Normal}(0.5, 1^2) \\
b &\sim \operatorname{Normal}(a, 2^2) \\
x &\sim \operatorname{Normal}(b, 0.5^2)
\end{aligned}
$$

We assume that `x` is data, i.e., an observed variable.
In our small language this model will be defined as

```julia
@mini_model function m(x)
    a ~ Normal(0.5, 1)
    b ~ Normal(a, 2)
    x ~ Normal(b, 0.5)
    return nothing
end
```



Specifically, we demand that

  - all observed variables are arguments of the program,
  - the model definition does not contain any control flow,
  - all variables are scalars, and
  - the function returns `nothing`.

First, we import some required packages:

```julia
using MacroTools, Distributions, Random, AbstractMCMC, MCMCChains
```




Before getting to the actual "compiler", we first build the data structure for the program trace.
A program trace for a probabilistic programming language needs to at least record the values of stochastic variables and their log-probabilities.

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




Internally, our probabilistic programming language works with two main functions:

  - `assume` for sampling unobserved variables and computing their log-probabilities, and
  - `observe` for computing log-probabilities of observed variables (but not sampling them).

For different inference algorithms we may have to use different sampling procedures and different log-probability computations.
For instance, in some cases we might want to sample all variables from their prior distributions and in other cases we might only want to compute the log-likelihood of the observations based on a given set of values for the unobserved variables.
Thus depending on the inference algorithm we want to use different `assume` and `observe` implementations.
We can achieve this by providing this `context` information as a function argument to `assume` and `observe`.

**Note:** *Although the context system in this tutorial is inspired by DynamicPPL, Turing's context system is much more complicated for flexibility and efficiency reasons.
Thus readers are advised to refer to the documentation of DynamicPPL and Turing for more detailed information about their context system.*

Here we can see the implementation of a sampler that draws values of unobserved variables from the prior and computes the log-probability for every variable.

```julia
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
```




Next we define the "compiler" for our simple programming language.
The term compiler is actually a bit misleading here since its only purpose is to transform the function definition in the `@mini_model` macro by

  - adding the context information (`context`) and the tracing data structure (`varinfo`) as additional arguments, and
  - replacing tildes with calls to `assume` and `observe`.

Afterwards, as usual the Julia compiler will just-in-time compile the model function when it is called.

The manipulation of Julia expressions is an advanced part of the Julia language.
The [Julia documentation](https://docs.julialang.org/en/v1/manual/metaprogramming/) provides an introduction to and more details about this so-called metaprogramming.

```julia
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
```




For inference, we make use of the [AbstractMCMC interface](https://turinglang.github.io/AbstractMCMC.jl/dev/).
It provides a default implementation of a `sample` function for sampling a Markov chain.
The default implementation already supports e.g. sampling of multiple chains in parallel, thinning of samples, or discarding initial samples.

The AbstractMCMC interface requires us to at least

  - define a model that is a subtype of `AbstractMCMC.AbstractModel`,
  - define a sampler that is a subtype of `AbstractMCMC.AbstractSampler`,
  - implement `AbstractMCMC.step` for our model and sampler.

Thus here we define a `MiniModel` model.
In this model we store the model function and the observed data.

```julia
struct MiniModel{F,D} <: AbstractMCMC.AbstractModel
    f::F
    data::D # a NamedTuple of all the data
end
```




In the Turing compiler, the model-specific `DynamicPPL.Model` is constructed automatically when calling the model function.
But for the sake of simplicity here we construct the model manually.

To illustrate probabilistic inference with our mini language we implement an extremely simplistic Random-Walk Metropolis-Hastings sampler.
We hard-code the proposal step as part of the sampler and only allow normal distributions with zero mean and fixed standard deviation.
The Metropolis-Hastings sampler in Turing is more flexible.

```julia
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
```




We need to define two `step` functions, one for the first step and the other for the following steps.
In the first step we sample values from the prior distributions and in the following steps we sample with the random-walk proposal.
The two functions are identified by the different arguments they take.

```julia
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
```




To make it easier to analyze the samples and compare them with results from Turing, additionally we define a version of `AbstractMCMC.bundle_samples` for our model and sampler that returns a `MCMCChains.Chains` object of samples.

```julia
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
```




Let us check how our mini probabilistic programming language works.
We define the probabilistic model:

```julia
@mini_model function m(x)
    a ~ Normal(0.5, 1)
    b ~ Normal(a, 2)
    x ~ Normal(b, 0.5)
    return nothing
end;
```




We perform inference with data `x = 3.0`:

```julia
sample(MiniModel(m, (x=3.0,)), MHSampler(), 1_000_000; chain_type=Chains)
```

```
Chains MCMC chain (1000000×2×1 Array{Float64, 3}):

Iterations        = 1:1:1000000
Number of chains  = 1
Samples per chain = 1000000
parameters        = a, b

Summary Statistics
  parameters      mean       std      mcse      ess_bulk      ess_tail     
 rh ⋯
      Symbol   Float64   Float64   Float64       Float64       Float64   Fl
oat ⋯

           a    0.9723    0.8970    0.0031    82344.1981   120870.8126    1
.00 ⋯
           b    2.8790    0.4877    0.0012   172253.9528   216170.0093    1
.00 ⋯
                                                               2 columns om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           a   -0.7866    0.3694    0.9703    1.5771    2.7271
           b    1.9196    2.5514    2.8785    3.2068    3.8351
```





We compare these results with Turing.

```julia
using Turing
using PDMats

@model function turing_m(x)
    a ~ Normal(0.5, 1)
    b ~ Normal(a, 2)
    x ~ Normal(b, 0.5)
    return nothing
end

sample(turing_m(3.0), MH(ScalMat(2, 1.0)), 1_000_000)
```

```
Chains MCMC chain (1000000×3×1 Array{Float64, 3}):

Iterations        = 1:1:1000000
Number of chains  = 1
Samples per chain = 1000000
Wall duration     = 4.88 seconds
Compute duration  = 4.88 seconds
parameters        = a, b
internals         = lp

Summary Statistics
  parameters      mean       std      mcse      ess_bulk      ess_tail     
 rh ⋯
      Symbol   Float64   Float64   Float64       Float64       Float64   Fl
oat ⋯

           a    0.9771    0.9014    0.0032    77735.6906   119185.4467    1
.00 ⋯
           b    2.8830    0.4878    0.0012   172872.9078   216961.1360    1
.00 ⋯
                                                               2 columns om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           a   -0.7820    0.3654    0.9753    1.5838    2.7517
           b    1.9243    2.5551    2.8829    3.2125    3.8380
```





As you can see, with our simple probabilistic programming language and custom samplers we get similar results as Turing.


## Appendix

These tutorials are a part of the TuringTutorials repository, found at: [https://github.com/TuringLang/TuringTutorials](https://github.com/TuringLang/TuringTutorials).

To locally run this tutorial, do the following commands:

```
using TuringTutorials
TuringTutorials.weave("14-minituring", "14_minituring.jmd")
```

Computer Information:

```
Julia Version 1.9.2
Commit e4ee485e909 (2023-07-05 09:39 UTC)
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver2)
  Threads: 1 on 16 virtual cores
Environment:
  JULIA_CPU_THREADS = 16
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/7aa0085e-79a4-45f3-a5bd-9743c91cf3da
  JULIA_IMAGE_THREADS = 1

```

Package Information:

```
Status `/cache/build/default-amdci4-0/julialang/turingtutorials/tutorials/14-minituring/Project.toml`
  [80f14c24] AbstractMCMC v4.4.2
  [31c24e10] Distributions v0.25.98
  [c7f686f2] MCMCChains v6.0.3
  [1914dd2f] MacroTools v0.5.10
  [90014a1f] PDMats v0.11.17
  [fce5fe82] Turing v0.26.2
  [9a3f8284] Random
```

And the full manifest:

```
Status `/cache/build/default-amdci4-0/julialang/turingtutorials/tutorials/14-minituring/Manifest.toml`
  [47edcb42] ADTypes v0.1.5
  [621f4979] AbstractFFTs v1.4.0
  [80f14c24] AbstractMCMC v4.4.2
⌅ [7a57a42e] AbstractPPL v0.5.4
  [1520ce14] AbstractTrees v0.4.4
  [79e6a3ab] Adapt v3.6.2
  [0bf59076] AdvancedHMC v0.4.6
  [5b7e9947] AdvancedMH v0.7.4
  [576499cb] AdvancedPS v0.4.3
  [b5ca4192] AdvancedVI v0.2.4
  [dce04be8] ArgCheck v2.3.0
  [4fba245c] ArrayInterface v7.4.11
  [a9b6321e] Atomix v0.1.0
  [13072b0f] AxisAlgorithms v1.0.1
  [39de3d68] AxisArrays v0.4.6
  [198e06fe] BangBang v0.3.39
  [9718e550] Baselet v0.1.1
⌅ [76274a88] Bijectors v0.12.8
  [fa961155] CEnum v0.4.2
  [49dc2e85] Calculus v0.5.1
  [082447d4] ChainRules v1.52.0
  [d360d2e6] ChainRulesCore v1.16.0
  [9e997f8a] ChangesOfVariables v0.1.8
  [861a8166] Combinatorics v1.0.2
  [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v4.7.0
  [a33af91c] CompositionsBase v0.1.2
  [88cd18e8] ConsoleProgressMonitor v0.1.2
  [187b0558] ConstructionBase v1.5.2
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.15.0
  [864edb3b] DataStructures v0.18.14
  [e2d170a0] DataValueInterfaces v1.0.0
  [244e2a9f] DefineSingletons v0.1.2
  [8bb1440f] DelimitedFiles v1.9.1
  [b429d917] DensityInterface v0.4.0
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
  [31c24e10] Distributions v0.25.98
  [ced4e74d] DistributionsAD v0.6.49
  [ffbed154] DocStringExtensions v0.9.3
  [fa6b7ba4] DualNumbers v0.6.8
⌃ [366bfd00] DynamicPPL v0.23.0
  [cad2338a] EllipticalSliceSampling v1.1.0
  [4e289a0a] EnumX v1.0.4
  [e2ba6199] ExprTools v0.1.9
  [7a1cc6ca] FFTW v1.7.1
⌅ [1a297f60] FillArrays v1.0.0
  [59287772] Formatting v0.4.2
  [f6369f11] ForwardDiff v0.10.35
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
⌃ [d9f16b24] Functors v0.4.4
  [46192b85] GPUArraysCore v0.1.5
  [34004b35] HypergeometricFunctions v0.3.18
  [22cec73e] InitialValues v0.3.1
  [505f98c9] InplaceOps v0.3.0
  [a98d9a8b] Interpolations v0.14.7
  [8197267c] IntervalSets v0.7.4
  [3587e190] InverseFunctions v0.1.10
  [41ab1584] InvertedIndices v1.3.0
  [92d709cd] IrrationalConstants v0.2.2
  [c8e1da08] IterTools v1.8.0
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.4.1
  [63c18a36] KernelAbstractions v0.9.6
  [5ab0869b] KernelDensity v0.6.7
  [929cbde3] LLVM v6.1.0
  [8ac3fa9e] LRUCache v1.4.1
  [b964fa9f] LaTeXStrings v1.3.0
  [50d2b5c4] Lazy v0.15.1
  [1d6d02ad] LeftChildRightSiblingTrees v0.2.0
  [6f1fad26] Libtask v0.8.6
  [6fdf6af0] LogDensityProblems v2.1.1
  [996a588d] LogDensityProblemsAD v1.5.0
  [2ab3a3ac] LogExpFunctions v0.3.24
  [e6f89c97] LoggingExtras v1.0.0
  [c7f686f2] MCMCChains v6.0.3
  [be115224] MCMCDiagnosticTools v0.3.4
  [e80e1ace] MLJModelInterface v1.8.0
  [1914dd2f] MacroTools v0.5.10
  [dbb5928d] MappedArrays v0.4.2
  [128add7d] MicroCollections v0.1.4
  [e1d29d7a] Missings v1.1.0
  [872c559c] NNlib v0.9.2
  [77ba4419] NaNMath v1.0.2
  [86f7a689] NamedArrays v0.9.8
  [c020b1a1] NaturalSort v1.0.0
  [6fe1bfb0] OffsetArrays v1.12.10
  [3bd65402] Optimisers v0.2.18
  [bac558e1] OrderedCollections v1.6.0
  [90014a1f] PDMats v0.11.17
  [aea7be01] PrecompileTools v1.1.2
  [21216c6a] Preferences v1.4.0
  [08abe8d2] PrettyTables v2.2.4
  [33c8b6b6] ProgressLogging v0.1.4
  [92933f4c] ProgressMeter v1.7.2
  [1fd47b50] QuadGK v2.8.2
  [74087812] Random123 v1.6.1
  [e6cf234a] RandomNumbers v1.5.3
  [b3c3ace0] RangeArrays v0.3.2
  [c84ed2f1] Ratios v0.4.5
  [c1ae055f] RealDot v0.1.0
  [3cdcf5f2] RecipesBase v1.3.4
  [731186ca] RecursiveArrayTools v2.38.5
  [189a3867] Reexport v1.2.2
  [ae029012] Requires v1.3.0
  [79098fc4] Rmath v0.7.1
  [f2b01f46] Roots v2.0.17
  [7e49a35a] RuntimeGeneratedFunctions v0.5.11
  [0bca4576] SciMLBase v1.93.0
  [c0aeaf25] SciMLOperators v0.3.3
  [30f210dd] ScientificTypesBase v3.0.0
  [efcf1570] Setfield v1.1.1
  [ce78b400] SimpleUnPack v1.1.0
  [a2af1166] SortingAlgorithms v1.1.1
  [276daf66] SpecialFunctions v2.3.0
  [171d559e] SplittablesBase v0.1.15
  [90137ffa] StaticArrays v1.6.0
  [1e83bf80] StaticArraysCore v1.4.0
  [64bff920] StatisticalTraits v3.2.0
  [82ae8749] StatsAPI v1.6.0
  [2913bbd2] StatsBase v0.34.0
  [4c63d2b9] StatsFuns v1.3.0
  [892a3eda] StringManipulation v0.3.0
  [09ab397b] StructArrays v0.6.15
  [2efcf032] SymbolicIndexingInterface v0.2.2
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.10.1
  [5d786b92] TerminalLoggers v0.1.7
  [9f7883ad] Tracker v0.2.26
  [28d57a85] Transducers v0.4.77
  [410a4b4d] Tricks v0.1.7
  [781d530d] TruncatedStacktraces v1.4.0
  [fce5fe82] Turing v0.26.2
  [013be700] UnsafeAtomics v0.2.1
  [d80eeb9a] UnsafeAtomicsLLVM v0.1.3
  [efce3f68] WoodburyMatrices v0.5.5
  [700de1a5] ZygoteRules v0.2.3
  [f5851436] FFTW_jll v3.3.10+0
  [1d5cc7b8] IntelOpenMP_jll v2023.1.0+0
  [dad2f222] LLVMExtra_jll v0.0.23+0
  [856f044c] MKL_jll v2023.1.0+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [f50d1b31] Rmath_jll v0.4.0+0
  [0dad84c5] ArgTools v1.1.1
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [ade2ca70] Dates
  [8ba89e20] Distributed
  [f43a241f] Downloads v1.6.0
  [7b1f6079] FileWatching
  [9fa8497b] Future
  [b77e0a4c] InteractiveUtils
  [4af54fe1] LazyArtifacts
  [b27032c2] LibCURL v0.6.3
  [76f85450] LibGit2
  [8f399da3] Libdl
  [37e2e46d] LinearAlgebra
  [56ddb016] Logging
  [d6f4376e] Markdown
  [a63ad114] Mmap
  [ca575930] NetworkOptions v1.2.0
  [44cfe95a] Pkg v1.9.2
  [de0858da] Printf
  [3fa0cd96] REPL
  [9a3f8284] Random
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization
  [1a1011a3] SharedArrays
  [6462fe0b] Sockets
  [2f01184e] SparseArrays
  [10745b16] Statistics v1.9.0
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll v1.0.5+0
  [deac9b47] LibCURL_jll v7.84.0+0
  [29816b5a] LibSSH2_jll v1.10.2+0
  [c8ffd9c3] MbedTLS_jll v2.28.2+0
  [14a3606d] MozillaCACerts_jll v2022.10.11
  [4536629a] OpenBLAS_jll v0.3.21+4
  [05823500] OpenLibm_jll v0.8.1+0
  [bea87d4a] SuiteSparse_jll v5.10.1+6
  [83775a58] Zlib_jll v1.2.13+0
  [8e850b90] libblastrampoline_jll v5.8.0+0
  [8e850ede] nghttp2_jll v1.48.0+0
  [3f19e933] p7zip_jll v17.4.0+0
Info Packages marked with ⌃ and ⌅ have new versions available, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
```

