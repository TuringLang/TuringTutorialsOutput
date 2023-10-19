---
title: "Automatic Differentiation"
permalink: "docs/using-turing/autodiff"
---


# Automatic Differentiation

## Switching AD Modes

Turing supports four automatic differentiation (AD) packages in the back end during sampling. The default AD backend is [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) for forward-mode AD. Three reverse-mode AD backends are also supported, namely [Tracker](https://github.com/FluxML/Tracker.jl), [Zygote](https://github.com/FluxML/Zygote.jl) and [ReverseDiff](https://github.com/JuliaDiff/ReverseDiff.jl). `Zygote` and `ReverseDiff` are supported optionally if explicitly loaded by the user with `using Zygote` or `using ReverseDiff` next to `using Turing`.

To switch between the different AD backends, one can call the function `Turing.setadbackend(backend_sym)`, where `backend_sym` can be `:forwarddiff` (`ForwardDiff`), `:tracker` (`Tracker`), `:zygote` (`Zygote`) or `:reversediff` (`ReverseDiff.jl`). When using `ReverseDiff`, to compile the tape only once and cache it for later use, the user has to call `Turing.setrdcache(true)`. However, note that the use of caching in certain types of models can lead to incorrect results and/or errors.
Compiled tapes should only be used if you are absolutely certain that the computation doesn't change between different executions of your model.
Thus, e.g., in the model definition and all im- and explicitly called functions in the model all loops should be of fixed size, and `if`-statements should consistently execute the same branches.
For instance, `if`-statements with conditions that can be determined at compile time or conditions that depend only on the data will always execute the same branches during sampling (if the data is constant throughout sampling and, e.g., no mini-batching is used).
However, `if`-statements that depend on the model parameters can take different branches during sampling; hence, the compiled tape might be incorrect.
Thus you must not use compiled tapes when your model makes decisions based on the model parameters, and you should be careful if you compute functions of parameters that those functions do not have branching which might cause them to execute different code for different values of the parameter.

## Compositional Sampling with Differing AD Modes

Turing supports intermixed automatic differentiation methods for different variable spaces. The snippet below shows using `ForwardDiff` to sample the mean (`m`) parameter and using the Tracker-based `TrackerAD` autodiff for the variance (`s`) parameter:

```julia
using Turing

# Define a simple Normal model with unknown mean and variance.
@model function gdemo(x, y)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    return y ~ Normal(m, sqrt(s²))
end

# Sample using Gibbs and varying autodiff backends.
c = sample(
    gdemo(1.5, 2),
    Gibbs(HMC{Turing.ForwardDiffAD{1}}(0.1, 5, :m), HMC{Turing.TrackerAD}(0.1, 5, :s²)),
    1000,
)
```

```
Chains MCMC chain (1000×3×1 Array{Float64, 3}):

Iterations        = 1:1:1000
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 3.52 seconds
Compute duration  = 3.52 seconds
parameters        = s², m
internals         = lp

Summary Statistics
  parameters      mean       std      mcse   ess_bulk   ess_tail      rhat 
  e ⋯
      Symbol   Float64   Float64   Float64    Float64    Float64   Float64 
    ⋯

          s²    1.8187    1.1318    0.0747   236.2615   362.7316    0.9996 
    ⋯
           m    1.2408    0.7543    0.0633   147.4358   180.0100    1.0006 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

          s²    0.5676    1.0513    1.5317    2.2023    4.8511
           m   -0.2294    0.7636    1.2082    1.7373    2.7031
```





Generally, `TrackerAD` is faster when sampling from variables of high dimensionality (greater than 20), and `ForwardDiffAD` is more efficient for lower-dimension variables. This functionality allows those who are performance sensitive to fine-tune their automatic differentiation for their specific models.

If the differentiation method is not specified in this way, Turing will default to using whatever the global AD backend is. Currently, this defaults to `ForwardDiff`.
