---
mathjax: true
title: "Using External Sampler"
permalink: "/docs/using-turing/external-samplers"
---


# Using External Samplers on Turing Models

`Turing` provides several wrapped samplers from external sampling libraries, e.g., HMC samplers from `AdvancedHMC`.
These wrappers allow new users to seamlessly sample statistical models without leaving `Turing`
However, these wrappers might only sometimes be complete, missing some functionality from the wrapped sampling library.
Moreover, users might want to use samplers currently not wrapped within `Turing`.

For these reasons, `Turing` also makes running external samplers on Turing models easy without any necessary modifications or wrapping!
Throughout, we will use a 10-dimensional Neal's funnel as a running example::

```julia
# Import libraries.
using Turing, Random, LinearAlgebra

d = 10
@model function funnel()
    θ ~ Truncated(Normal(0, 3), -3, 3)
    z ~ MvNormal(zeros(d - 1), exp(θ) * I)
    return x ~ MvNormal(z, I)
end
```

```
funnel (generic function with 2 methods)
```





Now we sample the model to generate some observations, which we can then condition on.

```julia
(; x) = rand(funnel() | (θ=0,))
model = funnel() | (; x);
```




Users can use any sampler algorithm to sample this model if it follows the `AbstractMCMC` API.
Before discussing how this is done in practice, giving a high-level description of the process is interesting.
Imagine that we created an instance of an external sampler that we will call `spl` such that `typeof(spl)<:AbstractMCMC.AbstractSampler`.
In order to avoid type ambiguity within Turing, at the moment it is necessary to declare `spl` as an external sampler to Turing `espl = externalsampler(spl)`, where `externalsampler(s::AbstractMCMC.AbstractSampler)` is a Turing function that types our external sampler adequately.

An excellent point to start to show how this is done in practice is by looking at the sampling library `AdvancedMH` ((`AdvancedMH`'s GitHub)[[https://github.com/TuringLang/AdvancedMH.jl]) for Metropolis-Hastings (MH) methods.
Let's say we want to use a random walk Metropolis-Hastings sampler without specifying the proposal distributions.
The code below constructs an MH sampler using a multivariate Gaussian distribution with zero mean and unit variance in `d` dimensions as a random walk proposal.

```julia
# Importing the sampling library
using AdvancedMH
rwmh = AdvancedMH.RWMH(d)
```

```
AdvancedMH.MetropolisHastings{AdvancedMH.RandomWalkProposal{false, Distribu
tions.ZeroMeanIsoNormal{Tuple{Base.OneTo{Int64}}}}}(AdvancedMH.RandomWalkPr
oposal{false, Distributions.ZeroMeanIsoNormal{Tuple{Base.OneTo{Int64}}}}(Ze
roMeanIsoNormal(
dim: 10
μ: Zeros(10)
Σ: [1.0 0.0 … 0.0 0.0; 0.0 1.0 … 0.0 0.0; … ; 0.0 0.0 … 1.0 0.0; 0.0 0.0 … 
0.0 1.0]
)
))
```





Sampling is then as easy as:

```julia
chain = sample(model, externalsampler(rwmh), 10_000)
```

```
Chains MCMC chain (10000×11×1 Array{Float64, 3}):

Iterations        = 1:1:10000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 1.35 seconds
Compute duration  = 1.35 seconds
parameters        = θ, z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9]
internals         = lp

Summary Statistics
  parameters      mean       std      mcse   ess_bulk   ess_tail      rhat 
  e ⋯
      Symbol   Float64   Float64   Float64    Float64    Float64   Float64 
    ⋯

           θ    0.0121    0.8650    0.1314    47.4626    86.6241    1.0210 
    ⋯
        z[1]   -0.5324    0.7682    0.0892    79.9027   142.8232    1.0157 
    ⋯
        z[2]    1.4455    0.9096    0.1159    65.5889   170.0869    1.0017 
    ⋯
        z[3]    0.0219    0.7492    0.0668   128.3552   146.8813    1.0227 
    ⋯
        z[4]    0.8745    0.7209    0.0684   108.6626   194.7976    1.0117 
    ⋯
        z[5]    0.2145    0.7323    0.0834    81.6500   224.5874    1.0042 
    ⋯
        z[6]   -0.9286    0.8387    0.0970    68.8226    62.2614    1.0112 
    ⋯
        z[7]    0.0349    0.6947    0.0546   153.0607   206.9673    1.0096 
    ⋯
        z[8]   -0.6871    0.7771    0.0929    70.7733    73.1976    1.0010 
    ⋯
        z[9]   -0.4666    0.7502    0.0709   107.7177   150.2779    1.0320 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           θ   -1.4451   -0.6533    0.1445    0.6074    1.5776
        z[1]   -2.0639   -1.0141   -0.5672    0.0405    0.8580
        z[2]   -0.1706    0.8554    1.3782    2.0158    3.3755
        z[3]   -1.6737   -0.4554    0.0411    0.5822    1.3927
        z[4]   -0.3173    0.3945    0.7433    1.3393    2.5053
        z[5]   -1.1105   -0.3785    0.1465    0.7622    1.6893
        z[6]   -2.7768   -1.4725   -0.8207   -0.3002    0.3896
        z[7]   -1.4561   -0.3197   -0.0252    0.4072    1.5296
        z[8]   -2.2797   -1.1979   -0.5583   -0.2634    0.8090
        z[9]   -1.9283   -1.0024   -0.4508    0.0157    1.0232
```





## Going beyond the Turing API

As previously mentioned, the Turing wrappers can often limit the capabilities of the sampling libraries they wrap.
`AdvancedHMC`[^1] ((`AdvancedHMC`'s GitHub)[https://github.com/TuringLang/AdvancedHMC.jl]) is a clear example of this. A common practice when performing HMC is to provide an initial guess for the mass matrix.
However, the native HMC sampler within Turing only allows the user to specify the type of the mass matrix despite the two options being possible within `AdvancedHMC`.
Thankfully, we can use Turing's support for external samplers to define an HMC sampler with a custom mass matrix in `AdvancedHMC` and then use it to sample our Turing model.

We will use the library `Pathfinder`[^2] ((`Pathfinder`'s GitHub)[https://github.com/mlcolab/Pathfinder.jl]) to construct our estimate of mass matrix.
`Pathfinder` is a variational inference algorithm that first finds the maximum a posteriori (MAP) estimate of a target posterior distribution and then uses the trace of the optimization to construct a sequence of multivariate normal approximations to the target distribution.
In this process, `Pathfinder` computes an estimate of the mass matrix the user can access.

The code below shows this can be done in practice.

```julia
using AdvancedHMC, Pathfinder
# Running pathfinder
draws = 1_000
result_multi = multipathfinder(model, draws; nruns=8)

# Estimating the metric
inv_metric = result_multi.pathfinder_results[1].fit_distribution.Σ
metric = DenseEuclideanMetric(Matrix(inv_metric))

# Creating an AdvancedHMC NUTS sampler with the custom metric.
n_adapts = 1000 # Number of adaptation steps
tap = 0.9 # Large target acceptance probability to deal with the funnel structure of the posterior
nuts = AdvancedHMC.NUTS(tap; metric=metric)

# Sample
chain = sample(model, externalsampler(nuts), 10_000; n_adapts=1_000)
```

```
Chains MCMC chain (10000×23×1 Array{Float64, 3}):

Iterations        = 1:1:10000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 7.86 seconds
Compute duration  = 7.86 seconds
parameters        = θ, z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, 
tree_depth, numerical_error, step_size, nom_step_size, is_adapt

Summary Statistics
  parameters      mean       std      mcse    ess_bulk    ess_tail      rha
t   ⋯
      Symbol   Float64   Float64   Float64     Float64     Float64   Float6
4   ⋯

           θ   -0.2492    1.0673    0.0270   1671.7391   1194.4760    1.000
5   ⋯
        z[1]   -0.4565    0.7999    0.0427    696.1796    223.1810    1.000
8   ⋯
        z[2]    1.2484    0.9075    0.0195   1738.0874   6160.8376    1.000
9   ⋯
        z[3]   -0.0604    0.6995    0.0169   1831.5252    227.1630    1.000
5   ⋯
        z[4]    0.7984    0.7731    0.0113   5020.0991   6377.8484    1.000
3   ⋯
        z[5]    0.1271    0.6803    0.0149   2271.3501   6945.0708    1.000
4   ⋯
        z[6]   -0.8429    0.7960    0.0160   2097.4126   6229.0701    1.000
8   ⋯
        z[7]    0.0519    0.6792    0.0112   3494.3136   6492.5774    1.001
5   ⋯
        z[8]   -0.6428    0.7585    0.0166   1713.6610    275.4449    1.000
2   ⋯
        z[9]   -0.5094    0.7169    0.0090   6606.0620   6527.5550    0.999
9   ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           θ   -2.5807   -0.9079   -0.1165    0.5004    1.5692
        z[1]   -2.1186   -0.9179   -0.4120    0.0161    1.0303
        z[2]   -0.1721    0.5453    1.1648    1.8331    3.2340
        z[3]   -1.5168   -0.4808   -0.0554    0.3698    1.3044
        z[4]   -0.4726    0.2400    0.6981    1.2735    2.4939
        z[5]   -1.1811   -0.3058    0.0980    0.5481    1.5113
        z[6]   -2.6191   -1.3479   -0.7414   -0.2526    0.4340
        z[7]   -1.3363   -0.3714    0.0424    0.4862    1.3870
        z[8]   -2.2896   -1.1139   -0.5708   -0.1191    0.6331
        z[9]   -2.0732   -0.9451   -0.4368   -0.0097    0.7970
```





## Using new inference methods

So far we have used Turing's support for external samplers to go beyond the capabilities of the wrappers.
We want to use this support to employ a sampler not supported within Turing's ecosystem yet.
We will use the recently developed Micro-Cannoncial Hamiltonian Monte Carlo (MCHMC) sampler to showcase this.
MCHMC[^3,^4] ((MCHMC's GitHub)[https://github.com/JaimeRZP/MicroCanonicalHMC.jl]) is HMC sampler that uses one single Hamiltonian energy level to explore the whole parameter space.
This is achieved by simulating the dynamics of a microcanonical Hamiltonian with an additional noise term to ensure ergodicity.

Using this as well as other inference methods outside the Turing ecosystem is as simple as executing the code shown below:

```julia
using MicroCanonicalHMC
# Create MCHMC sampler
n_adapts = 1_000 # adaptation steps
tev = 0.01 # target energy variance
mchmc = MCHMC(n_adapts, tev; adaptive=true)

# Sample
chain = sample(model, externalsampler(mchmc), 10_000)
```

```
Chains MCMC chain (10000×11×1 Array{Float64, 3}):

Iterations        = 1:1:10000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 1.38 seconds
Compute duration  = 1.38 seconds
parameters        = θ, z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9]
internals         = lp

Summary Statistics
  parameters      mean       std      mcse   ess_bulk   ess_tail      rhat 
  e ⋯
      Symbol   Float64   Float64   Float64    Float64    Float64   Float64 
    ⋯

           θ   -0.1569    0.9784    0.0598   269.6519   340.0034    1.0155 
    ⋯
        z[1]   -0.5781    0.7430    0.0405   348.2620   605.1068    1.0128 
    ⋯
        z[2]    1.3363    0.8683    0.0456   372.5555   780.0139    1.0060 
    ⋯
        z[3]   -0.0674    0.6713    0.0335   410.2338   550.3580    1.0045 
    ⋯
        z[4]    0.8433    0.7846    0.0423   360.3244   695.7552    1.0092 
    ⋯
        z[5]    0.0755    0.6543    0.0296   494.4099   756.8816    1.0012 
    ⋯
        z[6]   -0.8784    0.7778    0.0332   580.4709   900.7821    1.0006 
    ⋯
        z[7]    0.0237    0.6999    0.0313   507.0552   844.3985    1.0049 
    ⋯
        z[8]   -0.6734    0.7208    0.0353   437.0096   627.6718    1.0053 
    ⋯
        z[9]   -0.5416    0.6778    0.0286   584.2623   845.4894    1.0020 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           θ   -2.3225   -0.7729   -0.0732    0.5369    1.5595
        z[1]   -2.1870   -1.0472   -0.5105   -0.0661    0.7538
        z[2]   -0.1247    0.6855    1.2825    1.9083    3.1700
        z[3]   -1.4243   -0.4847   -0.0793    0.3526    1.3199
        z[4]   -0.4886    0.2799    0.7717    1.3260    2.5832
        z[5]   -1.2468   -0.3372    0.0697    0.4793    1.4003
        z[6]   -2.5755   -1.3640   -0.8194   -0.3205    0.4669
        z[7]   -1.4323   -0.4014    0.0350    0.4669    1.3771
        z[8]   -2.2817   -1.1065   -0.6062   -0.1848    0.5883
        z[9]   -2.0221   -0.9544   -0.4811   -0.0792    0.6479
```





The only requirement to work with `externalsampler` is that the provided `sampler` must implement the AbstractMCMC.jl-interface [INSERT LINK] for a `model` of type `AbstractMCMC.LogDensityModel` [INSERT LINK].

As previously stated, in order to use external sampling libraries within `Turing` they must follow the `AbstractMCMC` API.
In this section, we will briefly dwell on what this entails.
First and foremost, the sampler should be a subtype of `AbstractMCMC.AbstractSampler`.
Second, the stepping function of the MCMC algorithm must be made defined using `AbstractMCMC.step` and follow the structure below:

```
# First step
function AbstractMCMC.step{T<:AbstractMCMC.AbstractSampler}(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    spl::T;
    kwargs...,
)
    [...]
    return transition, sample
end

# N+1 step
function AbstractMCMC.step{T<:AbstractMCMC.AbstractSampler}(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::T,
    state;
    kwargs...,
) 
    [...]
    return transition, sample
end
```

There are several characteristics to note in these functions:

  - There must be two `step` functions:
    
      + A function that performs the first step and initializes the sampler.
      + A function that performs the following steps and takes an extra input, `state`, which carries the initialization information.

  - The functions must follow the displayed signatures.
  - The output of the functions must be a transition, the current state of the sampler, and a sample, what is saved to the MCMC chain.

The last requirement is that the transition must be structured with a field `θ`, which contains the values of the parameters of the model for said transition.
This allows `Turing` to seamlessly extract the parameter values at each step of the chain when bundling the chains.
Note that if the external sampler produces transitions that Turing cannot parse, the bundling of the samples will be different or fail.

For practical examples of how to adapt a sampling library to the `AbstractMCMC` interface, the readers can consult the following libraries:

  - [AdvancedMH](https://github.com/TuringLang/AdvancedMH.jl/blob/458a602ac32a8514a117d4c671396a9ba8acbdab/src/mh-core.jl#L73-L115)
  - [AdvancedHMC](https://github.com/TuringLang/AdvancedHMC.jl/blob/762e55f894d142495a41a6eba0eed9201da0a600/src/abstractmcmc.jl#L102-L170)
  - [MicroCanonicalHMC](https://github.com/JaimeRZP/MicroCanonicalHMC.jl/blob/master/src/abstractmcmc.jl)

# Refences

[^1]: Xu et al., [AdvancedHMC.jl: A robust, modular and efficient implementation of advanced HMC algorithms](http://proceedings.mlr.press/v118/xu20a/xu20a.pdf), 2019
[^2]: Zhang et al., [Pathfinder: Parallel quasi-Newton variational inference](https://arxiv.org/abs/2108.03782), 2021
[^3]: Robnik et al, [Microcanonical Hamiltonian Monte Carlo](https://arxiv.org/abs/2212.08549), 2022
[^4]: Robnik and Seljak, [Langevine Hamiltonian Monte Carlo](https://arxiv.org/abs/2303.18221), 2023
