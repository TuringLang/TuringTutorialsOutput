---
title: "Guide"
permalink: "/docs/using-turing/guide/"
---


# Guide

## Basics

### Introduction

A probabilistic program is Julia code wrapped in a `@model` macro. It can use arbitrary Julia code, but to ensure correctness of inference it should not have external effects or modify global state. Stack-allocated variables are safe, but mutable heap-allocated objects may lead to subtle bugs when using task copying. By default Libtask deepcopies `Array` and `Dict` objects when copying task to avoid bugs with data stored in mutable structure in Turing models.

To specify distributions of random variables, Turing programs should use the `~` notation:

`x ~ distr` where `x` is a symbol and `distr` is a distribution. If `x` is undefined in the model function, inside the probabilistic program, this puts a random variable named `x`, distributed according to `distr`, in the current scope. `distr` can be a value of any type that implements `rand(distr)`, which samples a value from the distribution `distr`. If `x` is defined, this is used for conditioning in a style similar to [Anglican](https://probprog.github.io/anglican/index.html) (another PPL). In this case, `x` is an observed value, assumed to have been drawn from the distribution `distr`. The likelihood is computed using `logpdf(distr,y)`. The observe statements should be arranged so that every possible run traverses all of them in exactly the same order. This is equivalent to demanding that they are not placed inside stochastic control flow.

Available inference methods include Importance Sampling (IS), Sequential Monte Carlo (SMC), Particle Gibbs (PG), Hamiltonian Monte Carlo (HMC), Hamiltonian Monte Carlo with Dual Averaging (HMCDA) and The No-U-Turn Sampler (NUTS).

### Simple Gaussian Demo

Below is a simple Gaussian demo illustrate the basic usage of Turing.jl.

```julia
# Import packages.
using Turing
using StatsPlots

# Define a simple Normal model with unknown mean and variance.
@model function gdemo(x, y)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    return y ~ Normal(m, sqrt(s²))
end
```

```
gdemo (generic function with 2 methods)
```





Note: As a sanity check, the prior expectation of `s²` is `mean(InverseGamma(2, 3)) = 3/(2 - 1) = 3` and the prior expectation of `m` is 0. This can be easily checked using `Prior`:

```julia
p1 = sample(gdemo(missing, missing), Prior(), 100000)
```

```
Chains MCMC chain (100000×5×1 Array{Float64, 3}):

Iterations        = 1:1:100000
Number of chains  = 1
Samples per chain = 100000
Wall duration     = 1.68 seconds
Compute duration  = 1.68 seconds
parameters        = s², m, x, y
internals         = lp

Summary Statistics
  parameters      mean       std      mcse     ess_bulk     ess_tail      r
hat ⋯
      Symbol   Float64   Float64   Float64      Float64      Float64   Floa
t64 ⋯

          s²    2.9728    6.7733    0.0215   99082.0926   97424.9876    1.0
000 ⋯
           m   -0.0002    1.7068    0.0055   97128.5961   99672.5295    1.0
000 ⋯
           x    0.0016    2.4341    0.0078   97596.4500   97755.5409    1.0
000 ⋯
           y    0.0001    2.4399    0.0077   99420.1652   98275.7220    1.0
000 ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

          s²    0.5375    1.1164    1.7932    3.1320   12.0203
           m   -3.3790   -0.9049    0.0078    0.9007    3.3762
           x   -4.7960   -1.2786    0.0068    1.2910    4.8080
           y   -4.8022   -1.2841    0.0002    1.2908    4.8054
```





We can perform inference by using the `sample` function, the first argument of which is our probabilistic program and the second of which is a sampler. More information on each sampler is located in the [API](%7B%7Bsite.baseurl%7D%7D/docs/library).

```julia
#  Run sampler, collect results.
c1 = sample(gdemo(1.5, 2), SMC(), 1000)
c2 = sample(gdemo(1.5, 2), PG(10), 1000)
c3 = sample(gdemo(1.5, 2), HMC(0.1, 5), 1000)
c4 = sample(gdemo(1.5, 2), Gibbs(PG(10, :m), HMC(0.1, 5, :s²)), 1000)
c5 = sample(gdemo(1.5, 2), HMCDA(0.15, 0.65), 1000)
c6 = sample(gdemo(1.5, 2), NUTS(0.65), 1000)
```

```
Chains MCMC chain (1000×14×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 1.49 seconds
Compute duration  = 1.49 seconds
parameters        = s², m
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, 
tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std      mcse   ess_bulk   ess_tail      rhat 
  e ⋯
      Symbol   Float64   Float64   Float64    Float64    Float64   Float64 
    ⋯

          s²    2.0397    1.7662    0.0859   531.3216   491.2213    1.0048 
    ⋯
           m    1.2301    0.8823    0.0422   485.3409   391.2064    1.0064 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

          s²    0.5988    1.0658    1.5238    2.4005    6.8234
           m   -0.4014    0.6941    1.2179    1.6803    3.1833
```





The `MCMCChains` module (which is re-exported by Turing) provides plotting tools for the `Chain` objects returned by a `sample` function. See the [MCMCChains](https://github.com/TuringLang/MCMCChains.jl) repository for more information on the suite of tools available for diagnosing MCMC chains.

```julia
# Summarise results
describe(c3)

# Plot results
plot(c3)
savefig("gdemo-plot.png")
```



The arguments for each sampler are:

  - SMC: number of particles.
  - PG: number of particles, number of iterations.
  - HMC: leapfrog step size, leapfrog step numbers.
  - Gibbs: component sampler 1, component sampler 2, ...
  - HMCDA: total leapfrog length, target accept ratio.
  - NUTS: number of adaptation steps (optional), target accept ratio.

For detailed information on the samplers, please review Turing.jl's [API](%7B%7Bsite.baseurl%7D%7D/docs/library) documentation.

### Modelling Syntax Explained

Using this syntax, a probabilistic model is defined in Turing. The model function generated by Turing can then be used to condition the model onto data. Subsequently, the sample function can be used to generate samples from the posterior distribution.

In the following example, the defined model is conditioned to the data (arg*1 = 1, arg*2 = 2) by passing (1, 2) to the model function.

```julia
@model function model_name(arg_1, arg_2)
    return ...
end
```



The conditioned model can then be passed onto the sample function to run posterior inference.

```julia
model_func = model_name(1, 2)
chn = sample(model_func, HMC(..)) # Perform inference by sampling using HMC.
```



The returned chain contains samples of the variables in the model.

```julia
var_1 = mean(chn[:var_1]) # Taking the mean of a variable named var_1.
```



The key (`:var_1`) can be a `Symbol` or a `String`. For example, to fetch `x[1]`, one can use `chn[Symbol("x[1]")]` or `chn["x[1]"]`.
If you want to retrieve all parameters associated with a specific symbol, you can use `group`. As an example, if you have the
parameters `"x[1]"`, `"x[2]"`, and `"x[3]"`, calling `group(chn, :x)` or `group(chn, "x")` will return a new chain with only `"x[1]"`, `"x[2]"`, and `"x[3]"`.

Turing does not have a declarative form. More generally, the order in which you place the lines of a `@model` macro matters. For example, the following example works:

```julia
# Define a simple Normal model with unknown mean and variance.
@model function model_function(y)
    s ~ Poisson(1)
    y ~ Normal(s, 1)
    return y
end

sample(model_function(10), SMC(), 100)
```

```
Chains MCMC chain (100×3×1 Array{Float64, 3}):

Log evidence      = -22.42198736156986
Iterations        = 1:1:100
Number of chains  = 1
Samples per chain = 100
Wall duration     = 1.57 seconds
Compute duration  = 1.57 seconds
parameters        = s
internals         = lp, weight

Summary Statistics
  parameters      mean       std      mcse   ess_bulk   ess_tail      rhat 
  e ⋯
      Symbol   Float64   Float64   Float64    Float64    Float64   Float64 
    ⋯

           s    4.0000    0.0000       NaN        NaN        NaN       NaN 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           s    4.0000    4.0000    4.0000    4.0000    4.0000
```





But if we switch the `s ~ Poisson(1)` and `y ~ Normal(s, 1)` lines, the model will no longer sample correctly:

```julia
# Define a simple Normal model with unknown mean and variance.
@model function model_function(y)
    y ~ Normal(s, 1)
    s ~ Poisson(1)
    return y
end

sample(model_function(10), SMC(), 100)
```



### Sampling Multiple Chains

Turing supports distributed and threaded parallel sampling. To do so, call `sample(model, sampler, parallel_type, n, n_chains)`, where `parallel_type` can be either `MCMCThreads()` or `MCMCDistributed()` for thread and parallel sampling, respectively.

Having multiple chains in the same object is valuable for evaluating convergence. Some diagnostic functions like `gelmandiag` require multiple chains.

If you do not want parallelism or are on an older version Julia, you can sample multiple chains with the `mapreduce` function:

```julia
# Replace num_chains below with however many chains you wish to sample.
chains = mapreduce(c -> sample(model_fun, sampler, 1000), chainscat, 1:num_chains)
```



The `chains` variable now contains a `Chains` object which can be indexed by chain. To pull out the first chain from the `chains` object, use `chains[:,:,1]`. The method is the same if you use either of the below parallel sampling methods.

#### Multithreaded sampling

If you wish to perform multithreaded sampling and are running Julia 1.3 or greater, you can call `sample` with the following signature:

```julia
using Turing

@model function gdemo(x)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))

    for i in eachindex(x)
        x[i] ~ Normal(m, sqrt(s²))
    end
end

model = gdemo([1.5, 2.0])

# Sample four chains using multiple threads, each with 1000 samples.
sample(model, NUTS(), MCMCThreads(), 1000, 4)
```



Be aware that Turing cannot add threads for you -- you must have started your Julia instance with multiple threads to experience any kind of parallelism. See the [Julia documentation](https://docs.julialang.org/en/v1/manual/parallel-computing/#man-multithreading-1) for details on how to achieve this.

#### Distributed sampling

To perform distributed sampling (using multiple processes), you must first import `Distributed`.

Process parallel sampling can be done like so:

```julia
# Load Distributed to add processes and the @everywhere macro.
using Distributed

# Load Turing.
using Turing

# Add four processes to use for sampling.
addprocs(4; exeflags="--project=$(Base.active_project())")

# Initialize everything on all the processes.
# Note: Make sure to do this after you've already loaded Turing,
#       so each process does not have to precompile.
#       Parallel sampling may fail silently if you do not do this.
@everywhere using Turing

# Define a model on all processes.
@everywhere @model function gdemo(x)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))

    for i in eachindex(x)
        x[i] ~ Normal(m, sqrt(s²))
    end
end

# Declare the model instance everywhere.
@everywhere model = gdemo([1.5, 2.0])

# Sample four chains using multiple processes, each with 1000 samples.
sample(model, NUTS(), MCMCDistributed(), 1000, 4)
```



### Sampling from an Unconditional Distribution (The Prior)

Turing allows you to sample from a declared model's prior. If you wish to draw a chain from the prior to inspect your prior distributions, you can simply run

```julia
chain = sample(model, Prior(), n_samples)
```



You can also run your model (as if it were a function) from the prior distribution, by calling the model without specifying inputs or a sampler. In the below example, we specify a `gdemo` model which returns two variables, `x` and `y`. The model includes `x` and `y` as arguments, but calling the function without passing in `x` or `y` means that Turing's compiler will assume they are missing values to draw from the relevant distribution. The `return` statement is necessary to retrieve the sampled `x` and `y` values.
Assign the function with `missing` inputs to a variable, and Turing will produce a sample from the prior distribution.

```julia
@model function gdemo(x, y)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    y ~ Normal(m, sqrt(s²))
    return x, y
end
```

```
gdemo (generic function with 2 methods)
```





Assign the function with `missing` inputs to a variable, and Turing will produce a sample from the prior distribution.

```julia
# Samples from p(x,y)
g_prior_sample = gdemo(missing, missing)
g_prior_sample()
```

```
(1.424367069367373, 3.003479965217019)
```





### Sampling from a Conditional Distribution (The Posterior)

#### Treating observations as random variables

Inputs to the model that have a value `missing` are treated as parameters, aka random variables, to be estimated/sampled. This can be useful if you want to simulate draws for that parameter, or if you are sampling from a conditional distribution. Turing supports the following syntax:

```julia
@model function gdemo(x, ::Type{T}=Float64) where {T}
    if x === missing
        # Initialize `x` if missing
        x = Vector{T}(undef, 2)
    end
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    for i in eachindex(x)
        x[i] ~ Normal(m, sqrt(s²))
    end
end

# Construct a model with x = missing
model = gdemo(missing)
c = sample(model, HMC(0.01, 5), 500)
```

```
Chains MCMC chain (500×14×1 Array{Float64, 3}):

Iterations        = 1:1:500
Number of chains  = 1
Samples per chain = 500
Wall duration     = 1.45 seconds
Compute duration  = 1.45 seconds
parameters        = s², m, x[1], x[2]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, numerical_error, step_size, no
m_step_size

Summary Statistics
  parameters      mean       std      mcse   ess_bulk   ess_tail      rhat 
  e ⋯
      Symbol   Float64   Float64   Float64    Float64    Float64   Float64 
    ⋯

          s²    1.4841    0.5872    0.5196     1.3937    14.5628    1.9815 
    ⋯
           m    0.3016    0.7009    0.3406     5.4932    26.8660    1.5385 
    ⋯
        x[1]   -2.0716    0.2818    0.1486     3.7332    35.3529    1.2796 
    ⋯
        x[2]    1.4212    0.2639    0.2061     1.7507    25.3903    1.7222 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

          s²    0.7974    0.9800    1.2421    2.0632    2.5473
           m   -0.6914   -0.2862    0.2233    0.8933    1.5261
        x[1]   -2.5907   -2.2938   -2.0206   -1.8332   -1.6389
        x[2]    0.9478    1.2203    1.4080    1.6339    1.8582
```





Note the need to initialize `x` when missing since we are iterating over its elements later in the model. The generated values for `x` can be extracted from the `Chains` object using `c[:x]`.

Turing also supports mixed `missing` and non-`missing` values in `x`, where the missing ones will be treated as random variables to be sampled while the others get treated as observations. For example:

```julia
@model function gdemo(x)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    for i in eachindex(x)
        x[i] ~ Normal(m, sqrt(s²))
    end
end

# x[1] is a parameter, but x[2] is an observation
model = gdemo([missing, 2.4])
c = sample(model, HMC(0.01, 5), 500)
```

```
Chains MCMC chain (500×13×1 Array{Float64, 3}):

Iterations        = 1:1:500
Number of chains  = 1
Samples per chain = 500
Wall duration     = 1.42 seconds
Compute duration  = 1.42 seconds
parameters        = s², m, x[1]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, numerical_error, step_size, no
m_step_size

Summary Statistics
  parameters      mean       std      mcse   ess_bulk   ess_tail      rhat 
  e ⋯
      Symbol   Float64   Float64   Float64    Float64    Float64   Float64 
    ⋯

          s²    1.1230    0.3472    0.1335     6.9907    10.8719    1.0359 
    ⋯
           m    0.1349    0.6552    0.6013     1.3225    13.2450    2.1213 
    ⋯
        x[1]    0.5820    0.3162    0.2279     1.9218    10.9251    1.6032 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

          s²    0.4181    0.8808    1.0698    1.3711    1.7972
           m   -0.9705   -0.4591    0.1176    0.7942    1.1362
        x[1]    0.0183    0.3350    0.5956    0.7499    1.3055
```





#### Default Values

Arguments to Turing models can have default values much like how default values work in normal Julia functions. For instance, the following will assign `missing` to `x` and treat it as a random variable. If the default value is not `missing`, `x` will be assigned that value and will be treated as an observation instead.

```julia
using Turing

@model function generative(x=missing, ::Type{T}=Float64) where {T<:Real}
    if x === missing
        # Initialize x when missing
        x = Vector{T}(undef, 10)
    end
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    for i in 1:length(x)
        x[i] ~ Normal(m, sqrt(s²))
    end
    return s², m
end

m = generative()
chain = sample(m, HMC(0.01, 5), 1000)
```

```
Chains MCMC chain (1000×22×1 Array{Float64, 3}):

Iterations        = 1:1:1000
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 2.8 seconds
Compute duration  = 2.8 seconds
parameters        = s², m, x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], 
x[9], x[10]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, numerical_error, step_size, no
m_step_size

Summary Statistics
  parameters      mean       std      mcse   ess_bulk   ess_tail      rhat 
  e ⋯
      Symbol   Float64   Float64   Float64    Float64    Float64   Float64 
    ⋯

          s²    1.2103    0.4511    0.1629     8.8689    37.6435    1.0473 
    ⋯
           m    0.3292    0.3237    0.1471     5.4416    43.5081    1.1725 
    ⋯
        x[1]    0.4638    0.5204    0.1875     7.4796    12.5130    1.1440 
    ⋯
        x[2]   -0.7873    0.3293    0.0849    15.3988    35.6929    1.0061 
    ⋯
        x[3]    1.5962    0.7247    0.4604     2.6722    12.3062    1.8817 
    ⋯
        x[4]    0.9138    0.3870    0.1471     6.9785    26.7546    1.4994 
    ⋯
        x[5]    0.8944    0.2940    0.1019     9.0672    27.5149    1.1420 
    ⋯
        x[6]    1.7526    0.4939    0.2503     5.7895    27.0196    1.3247 
    ⋯
        x[7]   -0.8302    0.4666    0.1972     6.1731    21.8240    1.1059 
    ⋯
        x[8]    1.7535    0.8162    0.5297     2.5565    22.8628    2.0097 
    ⋯
        x[9]   -0.0266    0.2855    0.0967     9.3657    46.8213    1.0797 
    ⋯
       x[10]   -0.2617    0.6340    0.2868     6.6579    12.9094    1.3287 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

          s²    0.6697    0.8465    1.0449    1.5525    2.2243
           m   -0.2358    0.0968    0.2913    0.5521    0.9172
        x[1]   -0.4878    0.2462    0.3997    0.6444    1.7393
        x[2]   -1.4133   -1.0394   -0.7683   -0.5944   -0.1195
        x[3]    0.4012    0.9821    1.6638    2.2251    2.8739
        x[4]    0.1795    0.6453    0.8945    1.2279    1.5946
        x[5]    0.3176    0.6861    0.9226    1.1255    1.3966
        x[6]    0.7824    1.3397    1.9731    2.1355    2.4090
        x[7]   -1.7211   -1.2149   -0.7250   -0.4308   -0.1922
        x[8]    0.2539    1.0981    1.9474    2.3652    3.1192
        x[9]   -0.6308   -0.1768   -0.0199    0.1729    0.4617
       x[10]   -1.7023   -0.7197   -0.0673    0.2073    0.5731
```





#### Access Values inside Chain

You can access the values inside a chain several ways:

 1. Turn them into a `DataFrame` object
 2. Use their raw `AxisArray` form
 3. Create a three-dimensional `Array` object

For example, let `c` be a `Chain`:

 1. `DataFrame(c)` converts `c` to a `DataFrame`,
 2. `c.value` retrieves the values inside `c` as an `AxisArray`, and
 3. `c.value.data` retrieves the values inside `c` as a 3D `Array`.

#### Variable Types and Type Parameters

The element type of a vector (or matrix) of random variables should match the `eltype` of the its prior distribution, `<: Integer` for discrete distributions and `<: AbstractFloat` for continuous distributions. Moreover, if the continuous random variable is to be sampled using a Hamiltonian sampler, the vector's element type needs to either be:

 1. `Real` to enable auto-differentiation through the model which uses special number types that are sub-types of `Real`, or
 2. Some type parameter `T` defined in the model header using the type parameter syntax, e.g. `function gdemo(x, ::Type{T} = Float64) where {T}`.
    Similarly, when using a particle sampler, the Julia variable used should either be:
 3. An `Array`, or
 4. An instance of some type parameter `T` defined in the model header using the type parameter syntax, e.g. `function gdemo(x, ::Type{T} = Vector{Float64}) where {T}`.

### Querying Probabilities from Model or Chain

Consider first the following simplified `gdemo` model:

```julia
@model function gdemo0(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    return x ~ Normal(m, sqrt(s))
end

# Instantiate three models, with different value of x
model1 = gdemo0(1)
model4 = gdemo0(4)
model10 = gdemo0(10)
```

```
DynamicPPL.Model{typeof(Main.var"##WeaveSandBox#501".gdemo0), (:x,), (), ()
, Tuple{Int64}, Tuple{}, DynamicPPL.DefaultContext}(Main.var"##WeaveSandBox
#501".gdemo0, (x = 10,), NamedTuple(), DynamicPPL.DefaultContext())
```





Now, query the instantiated models: compute the likelihood of `x = 1.0` given the values of `s = 1.0` and `m = 1.0` for the parameters:

```julia
prob"x = 1.0 | model = model1, s = 1.0, m = 1.0"
```

```
0.39894228040143265
```



```julia
prob"x = 1.0 | model = model4, s = 1.0, m = 1.0"
```

```
0.39894228040143265
```



```julia
prob"x = 1.0 | model = model10, s = 1.0, m = 1.0"
```

```
0.39894228040143265
```





Notice that even if we use three models, instantiated with three different values of `x`, we should obtain the same likelihood. We can easily verify that value in this case:

```julia
pdf(Normal(1.0, 1.0), 1.0)
```

```
0.3989422804014327
```





Let us now consider the following `gdemo` model:

```julia
@model function gdemo(x, y)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    return y ~ Normal(m, sqrt(s²))
end

# Instantiate the model.
model = gdemo(2.0, 4.0)
```

```
DynamicPPL.Model{typeof(Main.var"##WeaveSandBox#501".gdemo), (:x, :y), (), 
(), Tuple{Float64, Float64}, Tuple{}, DynamicPPL.DefaultContext}(Main.var"#
#WeaveSandBox#501".gdemo, (x = 2.0, y = 4.0), NamedTuple(), DynamicPPL.Defa
ultContext())
```





The following are examples of valid queries of the `Turing` model or chain:

  - `prob"x = 1.0, y = 1.0 | model = model, s = 1.0, m = 1.0"` calculates the likelihood of `x = 1` and `y = 1` given `s = 1` and `m = 1`.

  - `prob"s² = 1.0, m = 1.0 | model = model, x = nothing, y = nothing"` calculates the joint probability of `s = 1` and `m = 1` ignoring `x` and `y`. `x` and `y` are ignored so they can be optionally dropped from the RHS of `|`, but it is recommended to define them.
  - `prob"s² = 1.0, m = 1.0, x = 1.0 | model = model, y = nothing"` calculates the joint probability of `s = 1`, `m = 1` and `x = 1` ignoring `y`.
  - `prob"s² = 1.0, m = 1.0, x = 1.0, y = 1.0 | model = model"` calculates the joint probability of all the variables.
  - After the MCMC sampling, given a `chain`, `prob"x = 1.0, y = 1.0 | chain = chain, model = model"` calculates the element-wise likelihood of `x = 1.0` and `y = 1.0` for each sample in `chain`.
  - If `save_state=true` was used during sampling (i.e., `sample(model, sampler, N; save_state=true)`), you can simply do `prob"x = 1.0, y = 1.0 | chain = chain"`.

In all the above cases, `logprob` can be used instead of `prob` to calculate the log probabilities instead.

### Maximum likelihood and maximum a posterior estimates

Turing provides support for two mode estimation techniques, [maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) (MLE) and [maximum a posterior](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) (MAP) estimation. Optimization is performed by the [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) package. Mode estimation is currently a optional tool, and will not be available to you unless you have manually installed Optim and loaded the package with a `using` statement. To install Optim, run `import Pkg; Pkg.add("Optim")`.

Mode estimation only works when all model parameters are continuous -- discrete parameters cannot be estimated with MLE/MAP as of yet.

To understand how mode estimation works, let us first load Turing and Optim to enable mode estimation, and then declare a model:

```julia
# Note that loading Optim explicitly is required for mode estimation to function,
# as Turing does not load the opimization suite unless Optim is loaded as well.
using Turing
using Optim

@model function gdemo(x)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))

    for i in eachindex(x)
        x[i] ~ Normal(m, sqrt(s²))
    end
end
```

```
gdemo (generic function with 6 methods)
```





Once the model is defined, we can construct a model instance as we normally would:

```julia
# Create some data to pass to the model.
data = [1.5, 2.0]

# Instantiate the gdemo model with our data.
model = gdemo(data)
```

```
DynamicPPL.Model{typeof(Main.var"##WeaveSandBox#501".gdemo), (:x,), (), (),
 Tuple{Vector{Float64}}, Tuple{}, DynamicPPL.DefaultContext}(Main.var"##Wea
veSandBox#501".gdemo, (x = [1.5, 2.0],), NamedTuple(), DynamicPPL.DefaultCo
ntext())
```





Mode estimation is typically quick and easy at this point. Turing extends the function `Optim.optimize` and accepts the structs `MLE()` or `MAP()`, which inform Turing whether to provide an MLE or MAP estimate, respectively. By default, the [LBFGS optimizer](https://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/) is used, though this can be changed. Basic usage is:

```julia
# Generate a MLE estimate.
mle_estimate = optimize(model, MLE())

# Generate a MAP estimate.
map_estimate = optimize(model, MAP())
```

```
ModeResult with maximized lp of -4.62
[0.9074074074074423, 1.1666666666667456]
```





If you wish to change to a different optimizer, such as `NelderMead`, simply place your optimizer in the third argument slot:

```julia
# Use NelderMead
mle_estimate = optimize(model, MLE(), NelderMead())

# Use SimulatedAnnealing
mle_estimate = optimize(model, MLE(), SimulatedAnnealing())

# Use ParticleSwarm
mle_estimate = optimize(model, MLE(), ParticleSwarm())

# Use Newton
mle_estimate = optimize(model, MLE(), Newton())

# Use AcceleratedGradientDescent
mle_estimate = optimize(model, MLE(), AcceleratedGradientDescent())
```



Some methods may have trouble calculating the mode because not enough iterations were allowed, or the target function moved upwards between function calls. Turing will warn you if Optim fails to converge by running `Optim.converge`. A typical solution to this might be to add more iterations, or allow the optimizer to increase between function iterations:

```julia
# Increase the iterations and allow function eval to increase between calls.
mle_estimate = optimize(
    model, MLE(), Newton(), Optim.Options(; iterations=10_000, allow_f_increases=true)
)
```



More options for Optim are available [here](https://julianlsolvers.github.io/Optim.jl/stable/#user/config/).

#### Analyzing your mode estimate

Turing extends several methods from `StatsBase` that can be used to analyze your mode estimation results. Methods implemented include `vcov`, `informationmatrix`, `coeftable`, `params`, and `coef`, among others.

For example, let's examine our ML estimate from above using `coeftable`:

```julia
# Import StatsBase to use it's statistical methods.
using StatsBase

# Print out the coefficient table.
coeftable(mle_estimate)
```



```
─────────────────────────────
   estimate  stderror   tstat
─────────────────────────────
s    0.0625  0.0625    1.0
m    1.75    0.176777  9.8995
─────────────────────────────
```

Standard errors are calculated from the Fisher information matrix (inverse Hessian of the log likelihood or log joint). t-statistics will be familiar to frequentist statisticians. Warning -- standard errors calculated in this way may not always be appropriate for MAP estimates, so please be cautious in interpreting them.

#### Sampling with the MAP/MLE as initial states

You can begin sampling your chain from an MLE/MAP estimate by extracting the vector of parameter values and providing it to the `sample` function with the keyword `init_params`. For example, here is how to sample from the full posterior using the MAP estimate as the starting point:

```julia
# Generate an MAP estimate.
map_estimate = optimize(model, MAP())

# Sample with the MAP estimate as the starting point.
chain = sample(model, NUTS(), 1_000; init_params=map_estimate.values.array)
```



## Beyond the Basics

### Compositional Sampling Using Gibbs

Turing.jl provides a Gibbs interface to combine different samplers. For example, one can combine an `HMC` sampler with a `PG` sampler to run inference for different parameters in a single model as below.

```julia
@model function simple_choice(xs)
    p ~ Beta(2, 2)
    z ~ Bernoulli(p)
    for i in 1:length(xs)
        if z == 1
            xs[i] ~ Normal(0, 1)
        else
            xs[i] ~ Normal(2, 1)
        end
    end
end

simple_choice_f = simple_choice([1.5, 2.0, 0.3])

chn = sample(simple_choice_f, Gibbs(HMC(0.2, 3, :p), PG(20, :z)), 1000)
```

```
Chains MCMC chain (1000×3×1 Array{Float64, 3}):

Iterations        = 1:1:1000
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 16.44 seconds
Compute duration  = 16.44 seconds
parameters        = p, z
internals         = lp

Summary Statistics
  parameters      mean       std      mcse   ess_bulk   ess_tail      rhat 
  e ⋯
      Symbol   Float64   Float64   Float64    Float64    Float64   Float64 
    ⋯

           p    0.4026    0.2068    0.0201   105.3306   131.0926    1.0046 
    ⋯
           z    0.1710    0.3767    0.0208   328.0747        NaN    1.0021 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           p    0.0661    0.2429    0.3803    0.5477    0.8590
           z    0.0000    0.0000    0.0000    0.0000    1.0000
```





The `Gibbs` sampler can be used to specify unique automatic differentiation backends for different variable spaces. Please see the [Automatic Differentiation](%7B%7Bsite.baseurl%7D%7D/docs/using-turing/autodiff) article for more.

For more details of compositional sampling in Turing.jl, please check the corresponding [paper](http://proceedings.mlr.press/v84/ge18b.html).

### Working with filldist and arraydist

Turing provides `filldist(dist::Distribution, n::Int)` and `arraydist(dists::AbstractVector{<:Distribution})` as a simplified interface to construct product distributions, e.g., to model a set of variables that share the same structure but vary by group.

#### Constructing product distributions with filldist

The function `filldist` provides a general interface to construct product distributions over distributions of the same type and parameterisation.
Note that, in contrast to the product distribution interface provided by Distributions.jl (`Product`), `filldist` supports product distributions over univariate or multivariate distributions.

Example usage:

```julia
@model function demo(x, g)
    k = length(unique(g))
    a ~ filldist(Exponential(), k) # = Product(fill(Exponential(), k))
    mu = a[g]
    return x .~ Normal.(mu)
end
```

```
demo (generic function with 2 methods)
```





#### Constructing product distributions with arraydist

The function `arraydist` provides a general interface to construct product distributions over distributions of varying type and parameterisation.
Note that in contrast to the product distribution interface provided by Distributions.jl (`Product`), `arraydist` supports product distributions over univariate or multivariate distributions.

Example usage:

```julia
@model function demo(x, g)
    k = length(unique(g))
    a ~ arraydist([Exponential(i) for i in 1:k])
    mu = a[g]
    return x .~ Normal.(mu)
end
```

```
demo (generic function with 2 methods)
```





### Working with MCMCChains.jl

Turing.jl wraps its samples using `MCMCChains.Chain` so that all the functions working for `MCMCChains.Chain` can be re-used in Turing.jl. Two typical functions are `MCMCChains.describe` and `MCMCChains.plot`, which can be used as follows for an obtained chain `chn`. For more information on `MCMCChains`, please see the [GitHub repository](https://github.com/TuringLang/MCMCChains.jl).

```julia
describe(chn) # Lists statistics of the samples.
plot(chn) # Plots statistics of the samples.
```

![](figures/using-turing-guide_35_1.png)



There are numerous functions in addition to `describe` and `plot` in the `MCMCChains` package, such as those used in convergence diagnostics. For more information on the package, please see the [GitHub repository](https://github.com/TuringLang/MCMCChains.jl).

### Changing Default Settings

Some of Turing.jl's default settings can be changed for better usage.

#### AD Chunk Size

ForwardDiff (Turing's default AD backend) uses forward-mode chunk-wise AD. The chunk size can be set manually by `setchunksize(new_chunk_size)`.

#### AD Backend

Turing supports four packages of automatic differentiation (AD) in the back end during sampling. The default AD backend is [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) for forward-mode AD. Three reverse-mode AD backends are also supported, namely [Tracker](https://github.com/FluxML/Tracker.jl), [Zygote](https://github.com/FluxML/Zygote.jl) and [ReverseDiff](https://github.com/JuliaDiff/ReverseDiff.jl). `Zygote` and `ReverseDiff` are supported optionally if explicitly loaded by the user with `using Zygote` or `using ReverseDiff` next to `using Turing`.

For more information on Turing's automatic differentiation backend, please see the [Automatic Differentiation](%7B%7Bsite.baseurl%7D%7D/docs/using-turing/autodiff) article.

#### Progress Logging

`Turing.jl` uses ProgressLogging.jl to log the progress of sampling. Progress
logging is enabled as default but might slow down inference. It can be turned on
or off by setting the keyword argument `progress` of `sample` to `true` or `false`, respectively. Moreover, you can enable or disable progress logging globally by calling `setprogress!(true)` or `setprogress!(false)`, respectively.

Turing uses heuristics to select an appropriate visualization backend. If you
use [Juno](https://junolab.org/), the progress is displayed with a
[progress bar in the Atom window](http://docs.junolab.org/latest/man/juno_frontend/#Progress-Meters-1).
For Jupyter notebooks the default backend is
[ConsoleProgressMonitor.jl](https://github.com/tkf/ConsoleProgressMonitor.jl).
In all other cases, progress logs are displayed with
[TerminalLoggers.jl](https://github.com/c42f/TerminalLoggers.jl). Alternatively,
if you provide a custom visualization backend, Turing uses it instead of the
default backend.
