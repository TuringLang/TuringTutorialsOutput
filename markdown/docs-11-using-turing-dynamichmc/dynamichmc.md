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
chn = sample(gdemo(1.5, 2.0), DynamicNUTS(), 2000)
```

```
Chains MCMC chain (2000×3×1 Array{Float64, 3}):

Iterations        = 1:1:2000
Number of chains  = 1
Samples per chain = 2000
Wall duration     = 2.1 seconds
Compute duration  = 2.1 seconds
parameters        = s², m
internals         = lp

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat 
  e ⋯
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64 
    ⋯

          s²    2.0541    1.7454     0.0390    0.0544   919.4794    1.0013 
    ⋯
           m    1.1430    0.8196     0.0183    0.0253   908.8085    1.0007 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

          s²    0.5645    1.0298    1.5365    2.4190    7.0774
           m   -0.6683    0.6577    1.1683    1.6306    2.8000
```


