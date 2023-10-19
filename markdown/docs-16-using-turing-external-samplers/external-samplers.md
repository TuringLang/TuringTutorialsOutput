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

           θ    0.6607    0.7939    0.0973    72.5961   105.7628    1.0213 
    ⋯
        z[1]   -0.8104    0.8644    0.0741   136.5748   193.1133    1.0280 
    ⋯
        z[2]    2.4779    0.9590    0.0881   117.4520   114.9107    1.0036 
    ⋯
        z[3]   -0.4896    0.8236    0.0665   152.8934   324.6177    1.0092 
    ⋯
        z[4]    1.0283    0.8642    0.0719   147.8005   288.1243    1.0117 
    ⋯
        z[5]   -0.4502    0.8203    0.0575   207.4025   251.0889    1.0001 
    ⋯
        z[6]   -1.3978    0.8848    0.0744   145.0663   238.2676    1.0202 
    ⋯
        z[7]    0.9873    0.9029    0.0942    90.1870    66.8746    1.0043 
    ⋯
        z[8]   -0.8703    0.8725    0.0848   109.5568   247.2287    1.0175 
    ⋯
        z[9]   -0.2816    0.8113    0.0645   159.8647   285.8905    1.0184 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           θ   -0.8824    0.0585    0.7337    1.2290    2.0544
        z[1]   -2.5771   -1.4016   -0.7728   -0.2245    0.8919
        z[2]    0.7082    1.7266    2.5025    3.1241    4.3990
        z[3]   -2.1684   -1.0482   -0.3724    0.1027    0.9501
        z[4]   -0.5811    0.3906    1.0485    1.6438    2.7756
        z[5]   -2.1368   -1.0558   -0.3446    0.0909    1.0409
        z[6]   -3.1274   -1.9953   -1.3735   -0.6384    0.0386
        z[7]   -0.5023    0.3727    0.9843    1.5536    2.8396
        z[8]   -2.4849   -1.4396   -0.9295   -0.2852    0.7471
        z[9]   -1.9634   -0.8300   -0.2675    0.1937    1.3660
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
Wall duration     = 6.26 seconds
Compute duration  = 6.26 seconds
parameters        = θ, z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, 
tree_depth, numerical_error, step_size, nom_step_size, is_adapt

Summary Statistics
  parameters      mean       std      mcse     ess_bulk    ess_tail      rh
at  ⋯
      Symbol   Float64   Float64   Float64      Float64     Float64   Float
64  ⋯

           θ    0.7385    0.7657    0.0126    4301.7002   3078.2772    1.00
04  ⋯
        z[1]   -0.9031    0.8959    0.0370     692.9089    200.3252    1.00
22  ⋯
        z[2]    2.5099    1.1694    0.0742     593.4340    204.8295    1.00
32  ⋯
        z[3]   -0.4322    0.8423    0.0129    4056.1616   7074.6484    1.00
03  ⋯
        z[4]    0.9790    0.8442    0.0078   11919.2278   7939.1635    1.00
00  ⋯
        z[5]   -0.4131    0.7975    0.0095    6955.5167   7017.4933    1.00
05  ⋯
        z[6]   -1.4838    0.9516    0.0448     624.9735    223.4647    1.00
24  ⋯
        z[7]    1.0378    0.9231    0.0468     677.2989    210.1990    1.00
21  ⋯
        z[8]   -0.8099    0.8614    0.0314     826.6934    209.5013    1.00
22  ⋯
        z[9]   -0.3250    0.8579    0.0323     866.9676    215.1511    1.00
19  ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           θ   -0.9210    0.2811    0.7741    1.2525    2.1329
        z[1]   -2.6611   -1.4864   -0.8845   -0.3229    0.9355
        z[2]    0.3348    1.8182    2.5488    3.2617    4.5875
        z[3]   -2.1138   -0.9918   -0.4097    0.1444    1.1737
        z[4]   -0.6124    0.4035    0.9357    1.5376    2.7027
        z[5]   -2.0605   -0.9423   -0.3923    0.1390    1.1052
        z[6]   -3.3231   -2.1142   -1.4800   -0.8609    0.3563
        z[7]   -0.7578    0.4612    1.0240    1.6353    2.8166
        z[8]   -2.5228   -1.3693   -0.8053   -0.2413    1.0185
        z[9]   -2.0176   -0.8993   -0.3244    0.2207    1.5921
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
Wall duration     = 1.41 seconds
Compute duration  = 1.41 seconds
parameters        = θ, z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9]
internals         = lp

Summary Statistics
  parameters      mean       std      mcse    ess_bulk    ess_tail      rha
t   ⋯
      Symbol   Float64   Float64   Float64     Float64     Float64   Float6
4   ⋯

           θ    0.7663    0.7869    0.0403    434.9983    380.3436    1.000
4   ⋯
        z[1]   -0.9314    0.8360    0.0269    972.0595   2070.7483    1.002
0   ⋯
        z[2]    2.6098    1.0404    0.0406    658.0175    743.9614    1.000
3   ⋯
        z[3]   -0.4471    0.8165    0.0230   1272.3345   1735.5671    1.003
1   ⋯
        z[4]    0.9564    0.8298    0.0258   1032.6801   1920.9735    1.000
0   ⋯
        z[5]   -0.4248    0.8190    0.0253   1056.7992   1781.3167    1.003
0   ⋯
        z[6]   -1.5380    0.8654    0.0281    946.1267   2325.7645    1.000
2   ⋯
        z[7]    1.1074    0.8330    0.0312    737.7582   1763.6669    1.001
9   ⋯
        z[8]   -0.8519    0.8779    0.0292    929.0355   1818.5814    1.001
4   ⋯
        z[9]   -0.3862    0.8122    0.0256   1013.9282   1780.3013    1.000
2   ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           θ   -0.9543    0.3047    0.8214    1.2928    2.1504
        z[1]   -2.6378   -1.4718   -0.8928   -0.3625    0.6253
        z[2]    0.5581    1.9130    2.6086    3.3038    4.6596
        z[3]   -2.1266   -0.9705   -0.4364    0.1018    1.1283
        z[4]   -0.6325    0.3900    0.9388    1.4893    2.6484
        z[5]   -2.1151   -0.9563   -0.3915    0.1332    1.1057
        z[6]   -3.2846   -2.1265   -1.5133   -0.9230    0.0587
        z[7]   -0.4075    0.5168    1.0758    1.6585    2.8237
        z[8]   -2.6905   -1.3976   -0.8042   -0.2513    0.7669
        z[9]   -2.0031   -0.9190   -0.3692    0.1595    1.1766
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

The last requirement is that the transition must be structured with a field `θ` which contains the values of the parameters of the model for said transition.
This allows `Turing` to seamlessly extract the parameter values at each step of the chain when bundling the chains.
Note that if the external sampler produces transitions that Turing cannot parse the bundling of the samples will be different or fail.

For practical examples of how to adapt a sampling library to the `AbstractMCMC` interface, the readers can consult the following libraries:

  - (AdvancedMH)[https://github.com/TuringLang/AdvancedMH.jl/blob/458a602ac32a8514a117d4c671396a9ba8acbdab/src/mh-core.jl#L73-L115]
  - (AdvancedHMC)[https://github.com/TuringLang/AdvancedHMC.jl/blob/762e55f894d142495a41a6eba0eed9201da0a600/src/abstractmcmc.jl#L102-L170]
  - (MicroCanonicalHMC)[https://github.com/JaimeRZP/MicroCanonicalHMC.jl/blob/master/src/abstractmcmc.jl] within `MicroCanonicalHMC`.

# Refences

[^1]: Xu et al, (AdvancedHMC.jl: A robust, modular and efficient implementation of advanced HMC algorithms)[http://proceedings.mlr.press/v118/xu20a/xu20a.pdf], 2019
[^2]: Zhang et al, (Pathfinder: Parallel quasi-Newton variational inference)[https://arxiv.org/abs/2108.03782], 2021
[^3]: Robnik et al, (Microcanonical Hamiltonian Monte Carlo)[https://arxiv.org/abs/2212.08549], 2022
[^4]: Robnik and Seljak, (Langevine Hamiltonian Monte Carlo)[https://arxiv.org/abs/2303.18221], 2023
