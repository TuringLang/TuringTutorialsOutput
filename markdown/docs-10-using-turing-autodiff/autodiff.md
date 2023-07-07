---
title: "Automatic Differentiation"
permalink: "docs/using-turing/autodiff"
---


# Automatic Differentiation

## Switching AD Modes

Turing supports four packages of automatic differentiation (AD) in the back end during sampling. The default AD backend is [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) for forward-mode AD. Three reverse-mode AD backends are also supported, namely [Tracker](https://github.com/FluxML/Tracker.jl), [Zygote](https://github.com/FluxML/Zygote.jl) and [ReverseDiff](https://github.com/JuliaDiff/ReverseDiff.jl). `Zygote` and `ReverseDiff` are supported optionally if explicitly loaded by the user with `using Zygote` or `using ReverseDiff` next to `using Turing`.

To switch between the different AD backends, one can call function `Turing.setadbackend(backend_sym)`, where `backend_sym` can be `:forwarddiff` (`ForwardDiff`), `:tracker` (`Tracker`), `:zygote` (`Zygote`) or `:reversediff` (`ReverseDiff.jl`). When using `ReverseDiff`, to compile the tape only once and cache it for later use, the user has to call `Turing.setrdcache(true)`. However, note that the use of caching in certain types of models can lead to incorrect results and/or errors. Models for which the compiled tape can be safely cached are models with fixed size loops and no run-time if statements. Compile-time if statements are fine.

## Compositional Sampling with Differing AD Modes

Turing supports intermixed automatic differentiation methods for different variable spaces. The snippet below shows using `ForwardDiff` to sample the mean (`m`) parameter, and using the Tracker-based `TrackerAD` autodiff for the variance (`s`) parameter:

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
Wall duration     = 4.12 seconds
Compute duration  = 4.12 seconds
parameters        = s², m
internals         = lp

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat 
  e ⋯
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64 
    ⋯

          s²    2.0397    1.8600     0.0588    0.1568   130.1844    1.0017 
    ⋯
           m    1.0841    0.7823     0.0247    0.0761    59.2630    1.0005 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

          s²    0.5807    0.9830    1.4813    2.3697    6.6809
           m   -0.7629    0.6414    1.1108    1.5872    2.5070
```





Generally, `TrackerAD` is faster when sampling from variables of high dimensionality (greater than 20) and `ForwardDiffAD` is more efficient for lower-dimension variables. This functionality allows those who are performance sensitive to fine tune their automatic differentiation for their specific models.

If the differentiation method is not specified in this way, Turing will default to using whatever the global AD backend is. Currently, this defaults to `ForwardDiff`.
