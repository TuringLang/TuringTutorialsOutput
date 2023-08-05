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
Wall duration     = 1.94 seconds
Compute duration  = 1.94 seconds
parameters        = s², m
internals         = lp

Summary Statistics
  parameters      mean       std      mcse   ess_bulk   ess_tail      rhat 
  e ⋯
      Symbol   Float64   Float64   Float64    Float64    Float64   Float64 
    ⋯

          s²    1.9975    1.6819    0.0704   710.9798   697.1798    1.0009 
    ⋯
           m    1.1068    0.8278    0.0291   987.9563   605.1508    1.0005 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

          s²    0.5487    1.0092    1.5190    2.3996    6.2442
           m   -0.6053    0.6341    1.1485    1.6205    2.5705
```


