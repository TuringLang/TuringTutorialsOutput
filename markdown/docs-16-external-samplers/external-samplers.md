---
mathjax: true
title: "Using External Sampler"
permalink: "/tutorials/:name/"
---


# Using External Samplers

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

           θ    0.1251    0.7360    0.1086    45.7313    74.3627    1.0483 
    ⋯
        z[1]    1.0334    0.7854    0.0915    77.5902    77.0255    1.0269 
    ⋯
        z[2]    1.6867    0.9441    0.1180    65.8731    92.9540    1.0663 
    ⋯
        z[3]   -0.0435    0.6939    0.0667   108.4432   139.2407    1.0011 
    ⋯
        z[4]   -0.8929    0.7647    0.0800    90.3785   138.0823    1.0316 
    ⋯
        z[5]   -0.4822    0.7445    0.0763    96.4360   165.9024    1.0022 
    ⋯
        z[6]    0.2369    0.7198    0.0763    90.9533   155.6677    1.1273 
    ⋯
        z[7]   -0.1455    0.6872    0.0634   118.4996   153.2184    1.0047 
    ⋯
        z[8]    0.1315    0.7558    0.0731   105.0328   199.7070    1.0105 
    ⋯
        z[9]    0.6966    0.7387    0.0749    95.8488   148.8325    1.0017 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           θ   -1.1631   -0.4096    0.0995    0.6103    1.7149
        z[1]   -0.3408    0.3944    0.9593    1.4577    2.8949
        z[2]   -0.1032    1.0233    1.7073    2.3823    3.5842
        z[3]   -1.4138   -0.4395   -0.0741    0.4806    1.2862
        z[4]   -2.3217   -1.4061   -0.8761   -0.3751    0.5810
        z[5]   -2.1486   -1.0484   -0.4488    0.0636    0.7377
        z[6]   -1.2289   -0.3275    0.2678    0.7753    1.6268
        z[7]   -1.6858   -0.5670   -0.1335    0.2703    1.2280
        z[8]   -1.2469   -0.3692    0.1066    0.6343    1.6321
        z[9]   -0.7363    0.2114    0.7083    1.1218    2.2226
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
Wall duration     = 6.98 seconds
Compute duration  = 6.98 seconds
parameters        = θ, z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, 
tree_depth, numerical_error, step_size, nom_step_size, is_adapt

Summary Statistics
  parameters      mean       std      mcse     ess_bulk    ess_tail      rh
at  ⋯
      Symbol   Float64   Float64   Float64      Float64     Float64   Float
64  ⋯

           θ   -0.2176    1.0548    0.0282    1496.4249    907.0418    1.00
11  ⋯
        z[1]    0.9006    0.8639    0.0424     607.9910    219.8005    1.00
17  ⋯
        z[2]    1.3971    0.9360    0.0202    2145.2360   6944.8407    1.00
01  ⋯
        z[3]   -0.0665    0.6859    0.0164    1939.5876    991.7788    1.00
20  ⋯
        z[4]   -0.7894    0.7637    0.0110    4713.8293   5983.0116    1.00
00  ⋯
        z[5]   -0.4141    0.6928    0.0073    9192.1538   6091.4545    1.00
08  ⋯
        z[6]    0.3221    0.7062    0.0073    9073.0957   6808.6286    1.00
02  ⋯
        z[7]   -0.2132    0.6832    0.0065   11200.1436   6591.0970    1.00
03  ⋯
        z[8]    0.0446    0.6664    0.0070    7981.6021   6623.7699    1.00
03  ⋯
        z[9]    0.6444    0.7520    0.0103    5526.8992   5987.6958    1.00
01  ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           θ   -2.5503   -0.8764   -0.0889    0.5385    1.5423
        z[1]   -0.6300    0.3250    0.8411    1.4494    2.6704
        z[2]   -0.0943    0.6550    1.3178    2.0318    3.3726
        z[3]   -1.4388   -0.4723   -0.0532    0.3572    1.2759
        z[4]   -2.4679   -1.2839   -0.7075   -0.2434    0.4843
        z[5]   -1.9140   -0.8336   -0.3575    0.0302    0.8562
        z[6]   -1.0161   -0.1310    0.2754    0.7700    1.8067
        z[7]   -1.6644   -0.6149   -0.1909    0.2217    1.1043
        z[8]   -1.3095   -0.3642    0.0400    0.4638    1.3807
        z[9]   -0.6603    0.1063    0.5510    1.1085    2.3095
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
Wall duration     = 1.4 seconds
Compute duration  = 1.4 seconds
parameters        = θ, z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9]
internals         = lp

Summary Statistics
  parameters      mean       std      mcse   ess_bulk    ess_tail      rhat
    ⋯
      Symbol   Float64   Float64   Float64    Float64     Float64   Float64
    ⋯

           θ   -0.0365    0.9514    0.0610   250.1561    346.3037    1.0051
    ⋯
        z[1]    1.0572    0.7978    0.0528   244.2905    445.0157    1.0019
    ⋯
        z[2]    1.5332    0.9378    0.0475   381.9357   1007.4886    1.0020
    ⋯
        z[3]   -0.0472    0.6967    0.0299   545.1923   1037.9302    1.0004
    ⋯
        z[4]   -0.8469    0.7803    0.0382   427.4867    894.2087    1.0006
    ⋯
        z[5]   -0.3727    0.7150    0.0296   588.1250   1063.3397    1.0055
    ⋯
        z[6]    0.3693    0.6692    0.0350   372.8275    771.4557    1.0080
    ⋯
        z[7]   -0.2265    0.7333    0.0345   462.2421    706.4430    1.0005
    ⋯
        z[8]   -0.0121    0.6924    0.0285   598.2499    919.0822    1.0005
    ⋯
        z[9]    0.7059    0.7460    0.0331   544.6667    648.5883    1.0031
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           θ   -2.1771   -0.6035    0.0379    0.6018    1.6434
        z[1]   -0.3034    0.4622    0.9930    1.5904    2.7431
        z[2]   -0.0620    0.8390    1.4823    2.1328    3.5195
        z[3]   -1.4757   -0.4828   -0.0372    0.3902    1.3323
        z[4]   -2.4877   -1.3368   -0.8023   -0.2994    0.5478
        z[5]   -1.8689   -0.8222   -0.3324    0.0994    0.9799
        z[6]   -0.9380   -0.0678    0.3477    0.7952    1.7425
        z[7]   -1.7367   -0.6694   -0.1937    0.2257    1.2070
        z[8]   -1.4219   -0.4541   -0.0013    0.4244    1.3706
        z[9]   -0.6107    0.1975    0.6375    1.1535    2.3526
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
