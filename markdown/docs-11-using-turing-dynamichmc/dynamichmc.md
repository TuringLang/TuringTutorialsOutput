---
title: "Using DynamicHMC"
permalink: "/docs/using-turing/dynamichmc/"
---


# Using DynamicHMC

Turing supports the use of [DynamicHMC](https://github.com/tpapp/DynamicHMC.jl) as a sampler through the `DynamicNUTS` function.

To use the `DynamicNUTS` function, you must import the `DynamicHMC` package as well as Turing. Turing does not formally require `DynamicHMC` but will include additional functionality if both packages are present.

Here is a brief example of how to apply `DynamicNUTS`:

```julia
# Import Turing and DynamicHMC.
using DynamicHMC, Turing

# Model definition.
@model function gdemo(x, y)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    return y ~ Normal(m, sqrt(s²))
end

# Pull 2,000 samples using DynamicNUTS.
dynamic_nuts = externalsampler(DynamicHMC.NUTS())
chn = sample(gdemo(1.5, 2.0), dynamic_nuts, 2000)
```

```
Chains MCMC chain (2000×3×1 Array{Float64, 3}):

Iterations        = 1:1:2000
Number of chains  = 1
Samples per chain = 2000
Wall duration     = 1.92 seconds
Compute duration  = 1.92 seconds
parameters        = s², m
internals         = lp

Summary Statistics
  parameters      mean       std      mcse    ess_bulk   ess_tail      rhat
    ⋯
      Symbol   Float64   Float64   Float64     Float64    Float64   Float64
    ⋯

          s²    1.7975    1.3909    0.0492    916.7571   980.7918    1.0004
    ⋯
           m    1.1993    0.7259    0.0213   1205.1566   913.8278    0.9999
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

          s²    0.5864    1.0044    1.3953    2.0982    5.5052
           m   -0.2304    0.7316    1.1886    1.6468    2.6525
```


