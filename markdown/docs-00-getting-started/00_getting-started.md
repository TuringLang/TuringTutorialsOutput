---
redirect_from: "docs/0-gettingstarted/"
title: "Getting Started"
permalink: "/docs/using-turing/get-started"
---


# Getting Started

## Installation

To use Turing, you need to install Julia first and then install Turing.

### Install Julia

You will need to install Julia 1.3 or greater, which you can get from [the official Julia website](http://julialang.org/downloads/).

### Install Turing.jl

Turing is an officially registered Julia package, so you can install a stable version of Turing by running the following in the Julia REPL:

```julia
using Pkg
Pkg.add("Turing")
```




You can check if all tests pass by running `Pkg.test("Turing")` (it might take a long time)

## Example

Here's a simple example showing Turing in action.

First, we can load the Turing and StatsPlots modules

```julia
using Turing
using StatsPlots
```




Then, we define a simple Normal model with unknown mean and variance

```julia
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





Then we can run a sampler to collect results. In this case, it is a Hamiltonian Monte Carlo sampler

```julia
chn = sample(gdemo(1.5, 2), HMC(0.1, 5), 1000)
```

```
Chains MCMC chain (1000×11×1 Array{Float64, 3}):

Iterations        = 1:1:1000
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 1.31 seconds
Compute duration  = 1.31 seconds
parameters        = s², m
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat 
  e ⋯
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64 
    ⋯

          s²    2.2472    2.2512     0.0712    0.1609   143.0018    1.0090 
    ⋯
           m    1.0883    0.8516     0.0269    0.0815    56.0441    1.0291 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

          s²    0.5794    1.0258    1.5152    2.4543    8.0679
           m   -0.7367    0.5619    1.1344    1.5979    2.7766
```





We can plot the results

```julia
plot(chn)
```

![](figures/00_getting-started_5_1.png)



In this case, because we use the [normal-inverse gamma distribution](https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution)
as a [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior), we can compute
its updated mean as follows:

```julia
s² = InverseGamma(2, 3)
m = Normal(0, 1)
data = [1.5, 2]
x_bar = mean(data)
N = length(data)

mean_exp = (m.σ * m.μ + N * x_bar) / (m.σ + N)
```

```
1.1666666666666667
```





We can also compute the updated variance

```julia
updated_alpha = shape(s²) + (N / 2)
updated_beta =
    scale(s²) +
    (1 / 2) * sum((data[n] - x_bar)^2 for n in 1:N) +
    (N * m.σ) / (N + m.σ) * ((x_bar)^2) / 2
variance_exp = updated_beta / (updated_alpha - 1)
```

```
2.0416666666666665
```





Finally, we can check if these expectations align with our HMC approximations
from earlier. We can compute samples from a normal-inverse gamma following the
equations given [here](https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution#Generating_normal-inverse-gamma_random_variates).

```julia
function sample_posterior(alpha, beta, mean, lambda, iterations)
    samples = []
    for i in 1:iterations
        sample_variance = rand(InverseGamma(alpha, beta), 1)
        sample_x = rand(Normal(mean, sqrt(sample_variance[1]) / lambda), 1)
        sanples = append!(samples, sample_x)
    end
    return samples
end

analytical_samples = sample_posterior(updated_alpha, updated_beta, mean_exp, 2, 1000);
```


```julia
density(analytical_samples; label="Posterior (Analytical)")
density!(chn[:m]; label="Posterior (HMC)")
```

![](figures/00_getting-started_9_1.png)
