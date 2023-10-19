
# Import libraries.
using Turing, Random, LinearAlgebra

d = 10
@model function funnel()
    θ ~ Truncated(Normal(0, 3), -3, 3)
    z ~ MvNormal(zeros(d - 1), exp(θ) * I)
    return x ~ MvNormal(z, I)
end


(; x) = rand(funnel() | (θ=0,))
model = funnel() | (; x);


# Importing the sampling library
using AdvancedMH
rwmh = AdvancedMH.RWMH(d)


chain = sample(model, externalsampler(rwmh), 10_000)


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


using MicroCanonicalHMC
# Create MCHMC sampler
n_adapts = 1_000 # adaptation steps
tev = 0.01 # target energy variance
mchmc = MCHMC(n_adapts, tev; adaptive=true)

# Sample
chain = sample(model, externalsampler(mchmc), 10_000)

