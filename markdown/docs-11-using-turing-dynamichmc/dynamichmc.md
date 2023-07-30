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
Wall duration     = 1.98 seconds
Compute duration  = 1.98 seconds
parameters        = s², m
internals         = lp

Summary Statistics
  parameters      mean       std      mcse    ess_bulk   ess_tail      rhat
    ⋯
      Symbol   Float64   Float64   Float64     Float64    Float64   Float64
    ⋯

          s²    2.0152    1.7117    0.0635    798.4867   904.3355    1.0013
    ⋯
           m    1.1471    0.8258    0.0263   1094.2352   813.4655    1.0011
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

          s²    0.5752    1.0333    1.5093    2.3793    6.7551
           m   -0.4628    0.6476    1.1520    1.6284    2.7110
```


