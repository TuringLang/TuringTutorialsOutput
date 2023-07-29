---
redirect_from: "tutorials/9-variationalinference/"
title: "Variational inference (VI) in Turing.jl"
permalink: "/tutorials/:name/"
---


In this post we'll have a look at what's know as **variational inference (VI)**, a family of _approximate_ Bayesian inference methods, and how to use it in Turing.jl as an alternative to other approaches such as MCMC. In particular, we will focus on one of the more standard VI methods called **Automatic Differentation Variational Inference (ADVI)**.

Here we will focus on how to use VI in Turing and not much on the theory underlying VI.
If you are interested in understanding the mathematics you can checkout [our write-up](../../docs/for-developers/variational_inference) or any other resource online (there a lot of great ones).

Using VI in Turing.jl is very straight forward.
If `model` denotes a definition of a `Turing.Model`, performing VI is as simple as

```julia
m = model(data...) # instantiate model on the data
q = vi(m, vi_alg)  # perform VI on `m` using the VI method `vi_alg`, which returns a `VariationalPosterior`
```



Thus it's no more work than standard MCMC sampling in Turing.

To get a bit more into what we can do with `vi`, we'll first have a look at a simple example and then we'll reproduce the [tutorial on Bayesian linear regression](../../tutorials/5-linearregression) using VI instead of MCMC. Finally we'll look at some of the different parameters of `vi` and how you for example can use your own custom variational family.

We first import the packages to be used:

```julia
using Random
using Turing
using Turing: Variational
using StatsPlots, Measures

Random.seed!(42);
```




## Simple example: Normal-Gamma conjugate model

The Normal-(Inverse)Gamma conjugate model is defined by the following generative process

\begin{align}
s &\sim \mathrm{InverseGamma}(2, 3) \\
m &\sim \mathcal{N}(0, s) \\
x_i &\overset{\text{i.i.d.}}{=} \mathcal{N}(m, s), \quad i = 1, \dots, n
\end{align}

Recall that *conjugate* refers to the fact that we can obtain a closed-form expression for the posterior. Of course one wouldn't use something like variational inference for a conjugate model, but it's useful as a simple demonstration as we can compare the result to the true posterior.

First we generate some synthetic data, define the `Turing.Model` and instantiate the model on the data:

```julia
# generate data
x = randn(2000);
```


```julia
@model function model(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0.0, sqrt(s))
    for i in 1:length(x)
        x[i] ~ Normal(m, sqrt(s))
    end
end;
```


```julia
# Instantiate model
m = model(x);
```




Now we'll produce some samples from the posterior using a MCMC method, which in constrast to VI is guaranteed to converge to the *exact* posterior (as the number of samples go to infinity).

We'll produce 10 000 samples with 200 steps used for adaptation and a target acceptance rate of 0.65

If you don't understand what "adaptation" or "target acceptance rate" refers to, all you really need to know is that `NUTS` is known to be one of the most accurate and efficient samplers (when applicable) while requiring little to no hand-tuning to work well.

```julia
samples_nuts = sample(m, NUTS(), 10_000);
```




Now let's try VI. The most important function you need to now about to do VI in Turing is `vi`:

```julia
@doc(Variational.vi)
```


```
vi(model, alg::VariationalInference)
vi(model, alg::VariationalInference, q::VariationalPosterior)
vi(model, alg::VariationalInference, getq::Function, θ::AbstractArray)
```

Constructs the variational posterior from the `model` and performs the optimization following the configuration of the given `VariationalInference` instance.

# Arguments

  * `model`: `Turing.Model` or `Function` z ↦ log p(x, z) where `x` denotes the observations
  * `alg`: the VI algorithm used
  * `q`: a `VariationalPosterior` for which it is assumed a specialized implementation of the variational objective used exists.
  * `getq`: function taking parameters `θ` as input and returns a `VariationalPosterior`
  * `θ`: only required if `getq` is used, in which case it is the initial parameters for the variational posterior




Additionally, you can pass

  - an initial variational posterior `q`, for which we assume there exists a implementation of `update(::typeof(q), θ::AbstractVector)` returning an updated posterior `q` with parameters `θ`.
  - a function mapping $\theta \mapsto q_{\theta}$ (denoted above `getq`) together with initial parameters `θ`. This provides more flexibility in the types of variational families that we can use, and can sometimes be slightly more convenient for quick and rough work.

By default, i.e. when calling `vi(m, advi)`, Turing use a *mean-field* approximation with a multivariate normal as the base-distribution. Mean-field refers to the fact that we assume all the latent variables to be *independent*. This the "standard" ADVI approach; see [Automatic Differentiation Variational Inference (2016)](https://arxiv.org/abs/1603.00788) for more. In Turing, one can obtain such a mean-field approximation by calling `Variational.meanfield(model)` for which there exists an internal implementation for `update`:

```julia
@doc(Variational.meanfield)
```


```
meanfield([rng, ]model::Model)
```

Creates a mean-field approximation with multivariate normal as underlying distribution.




Currently the only implementation of `VariationalInference` available is `ADVI`, which is very convenient and applicable as long as your `Model` is differentiable with respect to the *variational parameters*, that is, the parameters of your variational distribution, e.g. mean and variance in the mean-field approximation.

```julia
@doc(Variational.ADVI)
```


```julia
struct ADVI{AD} <: AdvancedVI.VariationalInference{AD}
```

Automatic Differentiation Variational Inference (ADVI) with automatic differentiation backend `AD`.

# Fields

  * `samples_per_step::Int64`: Number of samples used to estimate the ELBO in each optimization step.
  * `max_iters::Int64`: Maximum number of gradient steps.




To perform VI on the model `m` using 10 samples for gradient estimation and taking 1000 gradient steps is then as simple as:

```julia
# ADVI
advi = ADVI(10, 1000)
q = vi(m, advi);
```




Unfortunately, for such a small problem Turing's new `NUTS` sampler is *so* efficient now that it's not that much more efficient to use ADVI. So, so very unfortunate...

With that being said, this is not the case in general. For very complex models we'll later find that `ADVI` produces very reasonable results in a much shorter time than `NUTS`.

And one significant advantage of using `vi` is that we can sample from the resulting `q` with ease. In fact, the result of the `vi` call is a `TransformedDistribution` from Bijectors.jl, and it implements the Distributions.jl interface for a `Distribution`:

```julia
q isa MultivariateDistribution
```

```
true
```





This means that we can call `rand` to sample from the variational posterior `q`

```julia
histogram(rand(q, 1_000)[1, :])
```

![](figures/09_variational-inference_12_1.png)



and `logpdf` to compute the log-probability

```julia
logpdf(q, rand(q))
```

```
5.091037132198697
```





Let's check the first and second moments of the data to see how our approximation compares to the point-estimates form the data:

```julia
var(x), mean(x)
```

```
(0.9785278394680044, -0.006560528402431126)
```



```julia
(mean(rand(q, 1000); dims=2)...,)
```

```
(0.9894666965134843, 0.0015528525885355482)
```






That's pretty close! But we're Bayesian so we're not interested in *just* matching the mean.
Let's instead look the actual density `q`.

For that we need samples:

```julia
samples = rand(q, 10000);
size(samples)
```

```
(2, 10000)
```



```julia
p1 = histogram(
    samples[1, :]; bins=100, normed=true, alpha=0.2, color=:blue, label="", ylabel="density"
)
density!(samples[1, :]; label="s (ADVI)", color=:blue, linewidth=2)
density!(samples_nuts, :s; label="s (NUTS)", color=:green, linewidth=2)
vline!([var(x)]; label="s (data)", color=:black)
vline!([mean(samples[1, :])]; color=:blue, label="")

p2 = histogram(
    samples[2, :]; bins=100, normed=true, alpha=0.2, color=:blue, label="", ylabel="density"
)
density!(samples[2, :]; label="m (ADVI)", color=:blue, linewidth=2)
density!(samples_nuts, :m; label="m (NUTS)", color=:green, linewidth=2)
vline!([mean(x)]; color=:black, label="m (data)")
vline!([mean(samples[2, :])]; color=:blue, label="")

plot(p1, p2; layout=(2, 1), size=(900, 500), legend=true)
```

![](figures/09_variational-inference_18_1.png)



For this particular `Model`, we can in fact obtain the posterior of the latent variables in closed form. This allows us to compare both `NUTS` and `ADVI` to the true posterior $p(s, m \mid x_1, \ldots, x_n)$.

*The code below is just work to get the marginals $p(s \mid x_1, \ldots, x_n)$ and $p(m \mid x_1, \ldots, x_n)$. Feel free to skip it.*

```julia
# closed form computation of the Normal-inverse-gamma posterior
# based on "Conjugate Bayesian analysis of the Gaussian distribution" by Murphy
function posterior(μ₀::Real, κ₀::Real, α₀::Real, β₀::Real, x::AbstractVector{<:Real})
    # Compute summary statistics
    n = length(x)
    x̄ = mean(x)
    sum_of_squares = sum(xi -> (xi - x̄)^2, x)

    # Compute parameters of the posterior
    κₙ = κ₀ + n
    μₙ = (κ₀ * μ₀ + n * x̄) / κₙ
    αₙ = α₀ + n / 2
    βₙ = β₀ + (sum_of_squares + n * κ₀ / κₙ * (x̄ - μ₀)^2) / 2

    return μₙ, κₙ, αₙ, βₙ
end
μₙ, κₙ, αₙ, βₙ = posterior(0.0, 1.0, 2.0, 3.0, x)

# marginal distribution of σ²
# cf. Eq. (90) in "Conjugate Bayesian analysis of the Gaussian distribution" by Murphy
p_σ² = InverseGamma(αₙ, βₙ)
p_σ²_pdf = z -> pdf(p_σ², z)

# marginal of μ
# Eq. (91) in "Conjugate Bayesian analysis of the Gaussian distribution" by Murphy
p_μ = μₙ + sqrt(βₙ / (αₙ * κₙ)) * TDist(2 * αₙ)
p_μ_pdf = z -> pdf(p_μ, z)

# posterior plots
p1 = plot()
histogram!(samples[1, :]; bins=100, normed=true, alpha=0.2, color=:blue, label="")
density!(samples[1, :]; label="s (ADVI)", color=:blue)
density!(samples_nuts, :s; label="s (NUTS)", color=:green)
vline!([mean(samples[1, :])]; linewidth=1.5, color=:blue, label="")
plot!(range(0.75, 1.35; length=1_001), p_σ²_pdf; label="s (posterior)", color=:red)
vline!([var(x)]; label="s (data)", linewidth=1.5, color=:black, alpha=0.7)
xlims!(0.75, 1.35)

p2 = plot()
histogram!(samples[2, :]; bins=100, normed=true, alpha=0.2, color=:blue, label="")
density!(samples[2, :]; label="m (ADVI)", color=:blue)
density!(samples_nuts, :m; label="m (NUTS)", color=:green)
vline!([mean(samples[2, :])]; linewidth=1.5, color=:blue, label="")
plot!(range(-0.25, 0.25; length=1_001), p_μ_pdf; label="m (posterior)", color=:red)
vline!([mean(x)]; label="m (data)", linewidth=1.5, color=:black, alpha=0.7)
xlims!(-0.25, 0.25)

plot(p1, p2; layout=(2, 1), size=(900, 500))
```

![](figures/09_variational-inference_19_1.png)



## Bayesian linear regression example using ADVI

This is simply a duplication of the tutorial [5. Linear regression](../regression/02_linear-regression) but now with the addition of an approximate posterior obtained using `ADVI`.

As we'll see, there is really no additional work required to apply variational inference to a more complex `Model`.

This section is basically copy-pasting the code from the [linear regression tutorial](../regression/02_linear-regression).

```julia
Random.seed!(1);
```


```julia
using FillArrays
using RDatasets

using LinearAlgebra
```


```julia
# Import the "Default" dataset.
data = RDatasets.dataset("datasets", "mtcars");

# Show the first six rows of the dataset.
first(data, 6)
```

```
6×12 DataFrame
 Row │ Model              MPG      Cyl    Disp     HP     DRat     WT      
 QS ⋯
     │ String31           Float64  Int64  Float64  Int64  Float64  Float64 
 Fl ⋯
─────┼─────────────────────────────────────────────────────────────────────
─────
   1 │ Mazda RX4             21.0      6    160.0    110     3.9     2.62  
    ⋯
   2 │ Mazda RX4 Wag         21.0      6    160.0    110     3.9     2.875
   3 │ Datsun 710            22.8      4    108.0     93     3.85    2.32
   4 │ Hornet 4 Drive        21.4      6    258.0    110     3.08    3.215
   5 │ Hornet Sportabout     18.7      8    360.0    175     3.15    3.44  
    ⋯
   6 │ Valiant               18.1      6    225.0    105     2.76    3.46
                                                               5 columns om
itted
```



```julia
# Function to split samples.
function split_data(df, at=0.70)
    r = size(df, 1)
    index = Int(round(r * at))
    train = df[1:index, :]
    test = df[(index + 1):end, :]
    return train, test
end

# A handy helper function to rescale our dataset.
function standardize(x)
    return (x .- mean(x; dims=1)) ./ std(x; dims=1)
end

function standardize(x, orig)
    return (x .- mean(orig; dims=1)) ./ std(orig; dims=1)
end

# Another helper function to unstandardize our datasets.
function unstandardize(x, orig)
    return x .* std(orig; dims=1) .+ mean(orig; dims=1)
end

function unstandardize(x, mean_train, std_train)
    return x .* std_train .+ mean_train
end
```

```
unstandardize (generic function with 2 methods)
```



```julia
# Remove the model column.
select!(data, Not(:Model))

# Split our dataset 70%/30% into training/test sets.
train, test = split_data(data, 0.7)
train_unstandardized = copy(train)

# Standardize both datasets.
std_train = standardize(Matrix(train))
std_test = standardize(Matrix(test), Matrix(train))

# Save dataframe versions of our dataset.
train_cut = DataFrame(std_train, names(data))
test_cut = DataFrame(std_test, names(data))

# Create our labels. These are the values we are trying to predict.
train_label = train_cut[:, :MPG]
test_label = test_cut[:, :MPG]

# Get the list of columns to keep.
remove_names = filter(x -> !in(x, [:MPG, :Model]), names(data))

# Filter the test and train sets.
train = Matrix(train_cut[:, remove_names]);
test = Matrix(test_cut[:, remove_names]);
```


```julia
# Bayesian linear regression.
@model function linear_regression(x, y, n_obs, n_vars, ::Type{T}=Vector{Float64}) where {T}
    # Set variance prior.
    σ² ~ truncated(Normal(0, 100), 0, Inf)

    # Set intercept prior.
    intercept ~ Normal(0, 3)

    # Set the priors on our coefficients.
    coefficients ~ MvNormal(Zeros(n_vars), 10.0 * I)

    # Calculate all the mu terms.
    mu = intercept .+ x * coefficients
    return y ~ MvNormal(mu, σ² * I)
end;
```


```julia
n_obs, n_vars = size(train)
m = linear_regression(train, train_label, n_obs, n_vars);
```




## Performing VI

First we define the initial variational distribution, or, equivalently, the family of distributions to consider. We're going to use the same mean-field approximation as Turing will use by default when we call `vi(m, advi)`, which we obtain by calling `Variational.meanfield`. This returns a `TransformedDistribution` with a `TuringDiagMvNormal` as the underlying distribution and the transformation mapping from the reals to the domain of the latent variables.

```julia
q0 = Variational.meanfield(m)
typeof(q0)
```

```
MultivariateTransformed{TuringDiagMvNormal{Vector{Float64}, Vector{Float64}
}, Stacked{Vector{Any}, Vector{UnitRange{Int64}}}} (alias for Bijectors.Tra
nsformedDistribution{DistributionsAD.TuringDiagMvNormal{Array{Float64, 1}, 
Array{Float64, 1}}, Bijectors.Stacked{Array{Any, 1}, Array{UnitRange{Int64}
, 1}}, Distributions.ArrayLikeVariate{1}})
```



```julia
advi = ADVI(10, 10_000)
```

```
AdvancedVI.ADVI{AdvancedVI.ForwardDiffAD{0}}(10, 10000)
```





Turing also provides a couple of different optimizers:

  - `TruncatedADAGrad` (default)
  - `DecayedADAGrad`
    as these are well-suited for problems with high-variance stochastic objectives, which is usually what the ELBO ends up being at different times in our optimization process.

With that being said, thanks to Requires.jl, if we add a `using Flux` prior to `using Turing` we can also make use of all the optimizers in `Flux`, e.g. `ADAM`, without any additional changes to your code! For example:

```julia
using Flux, Turing
using Turing.Variational

vi(m, advi; optimizer=Flux.ADAM())
```



just works.

For this problem we'll use the `DecayedADAGrad` from Turing:

```julia
opt = Variational.DecayedADAGrad(1e-2, 1.1, 0.9)
```

```
AdvancedVI.DecayedADAGrad(0.01, 1.1, 0.9, IdDict{Any, Any}())
```



```julia
q = vi(m, advi, q0; optimizer=opt)
typeof(q)
```

```
MultivariateTransformed{TuringDiagMvNormal{Vector{Float64}, Vector{Float64}
}, Stacked{Vector{Any}, Vector{UnitRange{Int64}}}} (alias for Bijectors.Tra
nsformedDistribution{DistributionsAD.TuringDiagMvNormal{Array{Float64, 1}, 
Array{Float64, 1}}, Bijectors.Stacked{Array{Any, 1}, Array{UnitRange{Int64}
, 1}}, Distributions.ArrayLikeVariate{1}})
```





*Note: as mentioned before, we internally define a `update(q::TransformedDistribution{<:TuringDiagMvNormal}, θ::AbstractVector)` method which takes in the current variational approximation `q` together with new parameters `z` and returns the new variational approximation. This is required so that we can actually update the `Distribution` object after each optimization step.*

*Alternatively, we can instead provide the mapping $\theta \mapsto q_{\theta}$ directly together with initial parameters using the signature `vi(m, advi, getq, θ_init)` as mentioned earlier. We'll see an explicit example of this later on!*

To compute statistics for our approximation we need samples:

```julia
z = rand(q, 10_000);
```




Now we can for example look at the average

```julia
avg = vec(mean(z; dims=2))
```

```
13-element Vector{Float64}:
  0.0005733072131103716
  0.0039810294562006845
  1.0005741750536699
 -1.3972955668198678e-5
 -0.0019714477109195523
 -0.0011259605434684032
  0.003869551279756123
  0.00021219560679366594
 -0.000935865845681252
 -0.0012422460281743964
 -0.002001693774264235
 -0.00071811873852161
 -0.0013369820613660516
```





The vector has the same ordering as the model, e.g. in this case `σ²` has index `1`, `intercept` has index `2` and `coefficients` has indices `3:12`. If  you forget or you might want to do something programmatically with the result, you can obtain the `sym → indices` mapping as follows:

```julia
_, sym2range = bijector(m, Val(true));
sym2range
```

```
(intercept = UnitRange{Int64}[2:2], σ² = UnitRange{Int64}[1:1], coefficient
s = UnitRange{Int64}[3:13])
```





For example, we can check the sample distribution and mean value of `σ²`:

```julia
histogram(z[1, :])
avg[union(sym2range[:σ²]...)]
```

```
1-element Vector{Float64}:
 0.0005733072131103716
```



```julia
avg[union(sym2range[:intercept]...)]
```

```
1-element Vector{Float64}:
 0.0039810294562006845
```



```julia
avg[union(sym2range[:coefficients]...)]
```

```
11-element Vector{Float64}:
  1.0005741750536699
 -1.3972955668198678e-5
 -0.0019714477109195523
 -0.0011259605434684032
  0.003869551279756123
  0.00021219560679366594
 -0.000935865845681252
 -0.0012422460281743964
 -0.002001693774264235
 -0.00071811873852161
 -0.0013369820613660516
```





*Note: as you can see, this is slightly awkward to work with at the moment. We'll soon add a better way of dealing with this.*

With a bit of work (this will be much easier in the future), we can also visualize the approximate marginals of the different variables, similar to `plot(chain)`:

```julia
function plot_variational_marginals(z, sym2range)
    ps = []

    for (i, sym) in enumerate(keys(sym2range))
        indices = union(sym2range[sym]...)  # <= array of ranges
        if sum(length.(indices)) > 1
            offset = 1
            for r in indices
                p = density(
                    z[r, :];
                    title="$(sym)[$offset]",
                    titlefontsize=10,
                    label="",
                    ylabel="Density",
                    margin=1.5mm,
                )
                push!(ps, p)
                offset += 1
            end
        else
            p = density(
                z[first(indices), :];
                title="$(sym)",
                titlefontsize=10,
                label="",
                ylabel="Density",
                margin=1.5mm,
            )
            push!(ps, p)
        end
    end

    return plot(ps...; layout=(length(ps), 1), size=(500, 2000), margin=4.0mm)
end
```

```
plot_variational_marginals (generic function with 1 method)
```



```julia
plot_variational_marginals(z, sym2range)
```

![](figures/09_variational-inference_39_1.png)



And let's compare this to using the `NUTS` sampler:

```julia
chain = sample(m, NUTS(), 10_000);
```


```julia
plot(chain; margin=12.00mm)
```

![](figures/09_variational-inference_41_1.png)

```julia
vi_mean = vec(mean(z; dims=2))[[
    union(sym2range[:coefficients]...)...,
    union(sym2range[:intercept]...)...,
    union(sym2range[:σ²]...)...,
]]
```

```
13-element Vector{Float64}:
  1.0005741750536699
 -1.3972955668198678e-5
 -0.0019714477109195523
 -0.0011259605434684032
  0.003869551279756123
  0.00021219560679366594
 -0.000935865845681252
 -0.0012422460281743964
 -0.002001693774264235
 -0.00071811873852161
 -0.0013369820613660516
  0.0039810294562006845
  0.0005733072131103716
```



```julia
mcmc_mean = mean(chain, names(chain, :parameters))[:, 2]
```

```
13-element Vector{Float64}:
  2.2859004223396092e-11
 -4.365634719043165e-9
  1.0000000096322366
  5.032160834778636e-8
  1.678694705786366e-7
  7.995072585215324e-9
 -3.050894541177501e-8
 -1.526192356929311e-7
  4.1906715229134646e-8
  2.639198917215915e-8
  3.979264843596925e-9
  8.211502841093717e-8
  1.7877535016488787e-8
```



```julia
plot(mcmc_mean; xticks=1:1:length(mcmc_mean), linestyle=:dot, label="NUTS")
plot!(vi_mean; linestyle=:dot, label="VI")
```

![](figures/09_variational-inference_44_1.png)



One thing we can look at is simply the squared error between the means:

```julia
sum(abs2, mcmc_mean .- vi_mean)
```

```
2.005136671712151
```





That looks pretty good! But let's see how the predictive distributions looks for the two.

## Prediction

Similarily to the linear regression tutorial, we're going to compare to multivariate ordinary linear regression using the `GLM` package:

```julia
# Import the GLM package.
using GLM

# Perform multivariate OLS.
ols = lm(
    @formula(MPG ~ Cyl + Disp + HP + DRat + WT + QSec + VS + AM + Gear + Carb), train_cut
)

# Store our predictions in the original dataframe.
train_cut.OLSPrediction = unstandardize(GLM.predict(ols), train_unstandardized.MPG)
test_cut.OLSPrediction = unstandardize(GLM.predict(ols, test_cut), train_unstandardized.MPG);
```


```julia
# Make a prediction given an input vector, using mean parameter values from a chain.
function prediction_chain(chain, x)
    p = get_params(chain)
    α = mean(p.intercept)
    β = collect(mean.(p.coefficients))
    return α .+ x * β
end
```

```
prediction_chain (generic function with 1 method)
```



```julia
# Make a prediction using samples from the variational posterior given an input vector.
function prediction(samples::AbstractVector, sym2ranges, x)
    α = mean(samples[union(sym2ranges[:intercept]...)])
    β = vec(mean(samples[union(sym2ranges[:coefficients]...)]; dims=2))
    return α .+ x * β
end

function prediction(samples::AbstractMatrix, sym2ranges, x)
    α = mean(samples[union(sym2ranges[:intercept]...), :])
    β = vec(mean(samples[union(sym2ranges[:coefficients]...), :]; dims=2))
    return α .+ x * β
end
```

```
prediction (generic function with 2 methods)
```



```julia
# Unstandardize the dependent variable.
train_cut.MPG = unstandardize(train_cut.MPG, train_unstandardized.MPG)
test_cut.MPG = unstandardize(test_cut.MPG, train_unstandardized.MPG);
```


```julia
# Show the first side rows of the modified dataframe.
first(test_cut, 6)
```

```
6×12 DataFrame
 Row │ MPG      Cyl       Disp       HP         DRat       WT          QSec
    ⋯
     │ Float64  Float64   Float64    Float64    Float64    Float64     Floa
t64 ⋯
─────┼─────────────────────────────────────────────────────────────────────
─────
   1 │    15.2   1.04746   0.565102   0.258882  -0.652405   0.0714991  -0.7
167 ⋯
   2 │    13.3   1.04746   0.929057   1.90345    0.380435   0.465717   -1.9
040
   3 │    19.2   1.04746   1.32466    0.691663  -0.777058   0.470584   -0.8
737
   4 │    27.3  -1.25696  -1.21511   -1.19526    1.0037    -1.38857     0.2
884
   5 │    26.0  -1.25696  -0.888346  -0.762482   1.62697   -1.18903    -1.0
936 ⋯
   6 │    30.4  -1.25696  -1.08773   -0.381634   0.451665  -1.79933    -0.9
680
                                                               6 columns om
itted
```



```julia
z = rand(q, 10_000);
```


```julia
# Calculate the predictions for the training and testing sets using the samples `z` from variational posterior
train_cut.VIPredictions = unstandardize(
    prediction(z, sym2range, train), train_unstandardized.MPG
)
test_cut.VIPredictions = unstandardize(
    prediction(z, sym2range, test), train_unstandardized.MPG
)

train_cut.BayesPredictions = unstandardize(
    prediction_chain(chain, train), train_unstandardized.MPG
)
test_cut.BayesPredictions = unstandardize(
    prediction_chain(chain, test), train_unstandardized.MPG
);
```


```julia
vi_loss1 = mean((train_cut.VIPredictions - train_cut.MPG) .^ 2)
bayes_loss1 = mean((train_cut.BayesPredictions - train_cut.MPG) .^ 2)
ols_loss1 = mean((train_cut.OLSPrediction - train_cut.MPG) .^ 2)

vi_loss2 = mean((test_cut.VIPredictions - test_cut.MPG) .^ 2)
bayes_loss2 = mean((test_cut.BayesPredictions - test_cut.MPG) .^ 2)
ols_loss2 = mean((test_cut.OLSPrediction - test_cut.MPG) .^ 2)

println("Training set:
    VI loss: $vi_loss1
    Bayes loss: $bayes_loss1
    OLS loss: $ols_loss1
Test set: 
    VI loss: $vi_loss2
    Bayes loss: $bayes_loss2
    OLS loss: $ols_loss2")
```

```
Training set:
    VI loss: 0.0013757680639403039
    Bayes loss: 9.181568032817367e-14
    OLS loss: 3.0709261248930093
Test set: 
    VI loss: 0.0014199510514292773
    Bayes loss: 1.3432106952596584e-12
    OLS loss: 27.09481307076057
```





Interestingly the squared difference between true- and mean-prediction on the test-set is actually *better* for the mean-field variational posterior than for the "true" posterior obtained by MCMC sampling using `NUTS`. But, as Bayesians, we know that the mean doesn't tell the entire story. One quick check is to look at the mean predictions ± standard deviation of the two different approaches:

```julia
z = rand(q, 1000);
preds = mapreduce(hcat, eachcol(z)) do zi
    return unstandardize(prediction(zi, sym2range, test), train_unstandardized.MPG)
end

scatter(
    1:size(test, 1),
    mean(preds; dims=2);
    yerr=std(preds; dims=2),
    label="prediction (mean ± std)",
    size=(900, 500),
    markersize=8,
)
scatter!(1:size(test, 1), unstandardize(test_label, train_unstandardized.MPG); label="true")
xaxis!(1:size(test, 1))
ylims!(10, 40)
title!("Mean-field ADVI (Normal)")
```

![](figures/09_variational-inference_54_1.png)

```julia
preds = mapreduce(hcat, 1:5:size(chain, 1)) do i
    return unstandardize(prediction_chain(chain[i], test), train_unstandardized.MPG)
end

scatter(
    1:size(test, 1),
    mean(preds; dims=2);
    yerr=std(preds; dims=2),
    label="prediction (mean ± std)",
    size=(900, 500),
    markersize=8,
)
scatter!(1:size(test, 1), unstandardize(test_label, train_unstandardized.MPG); label="true")
xaxis!(1:size(test, 1))
ylims!(10, 40)
title!("MCMC (NUTS)")
```

![](figures/09_variational-inference_55_1.png)



Indeed we see that the MCMC approach generally provides better uncertainty estimates than the mean-field ADVI approach! Good. So all the work we've done to make MCMC fast isn't for nothing.

## Alternative: provide parameter-to-distribution instead of $q$ with `update` implemented

As mentioned earlier, it's also possible to just provide the mapping $\theta \mapsto q_{\theta}$ rather than the variational family / initial variational posterior `q`, i.e. use the interface `vi(m, advi, getq, θ_init)` where `getq` is the mapping $\theta \mapsto q_{\theta}$

In this section we're going to construct a mean-field approximation to the model by hand using a composition of`Shift` and `Scale` from Bijectors.jl togheter with a standard multivariate Gaussian as the base distribution.

```julia
using Bijectors
```


```julia
using Bijectors: Scale, Shift
```


```julia
d = length(q)
base_dist = Turing.DistributionsAD.TuringDiagMvNormal(zeros(d), ones(d))
```

```
DistributionsAD.TuringDiagMvNormal{Vector{Float64}, Vector{Float64}}(
m: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
)
```





`bijector(model::Turing.Model)` is defined by Turing, and will return a `bijector` which takes you from the space of the latent variables to the real space. In this particular case, this is a mapping `((0, ∞) × ℝ × ℝ¹⁰) → ℝ¹²`. We're interested in using a normal distribution as a base-distribution and transform samples to the latent space, thus we need the inverse mapping from the reals to the latent space:

```julia
to_constrained = inverse(bijector(m));
```


```julia
function getq(θ)
    d = length(θ) ÷ 2
    A = @inbounds θ[1:d]
    b = @inbounds θ[(d + 1):(2 * d)]

    b = to_constrained ∘ Shift(b) ∘ Scale(exp.(A))

    return transformed(base_dist, b)
end
```

```
getq (generic function with 1 method)
```



```julia
q_mf_normal = vi(m, advi, getq, randn(2 * d));
```


```julia
p1 = plot_variational_marginals(rand(q_mf_normal, 10_000), sym2range) # MvDiagNormal + Affine transformation + to_constrained
p2 = plot_variational_marginals(rand(q, 10_000), sym2range)  # Turing.meanfield(m)

plot(p1, p2; layout=(1, 2), size=(800, 2000))
```

![](figures/09_variational-inference_62_1.png)



As expected, the fits look pretty much identical.

But using this interface it becomes trivial to go beyond the mean-field assumption we made for the variational posterior, as we'll see in the next section.

### Relaxing the mean-field assumption

Here we'll instead consider the variational family to be a full non-diagonal multivariate Gaussian. As in the previous section we'll implement this by transforming a standard multivariate Gaussian using `Scale` and `Shift`, but now `Scale` will instead be using a lower-triangular matrix (representing the Cholesky of the covariance matrix of a multivariate normal) in contrast to the diagonal matrix we used in for the mean-field approximate posterior.

```julia
# Using `ComponentArrays.jl` together with `UnPack.jl` makes our lives much easier.
using ComponentArrays, UnPack
```


```julia
proto_arr = ComponentArray(; L=zeros(d, d), b=zeros(d))
proto_axes = getaxes(proto_arr)
num_params = length(proto_arr)

function getq(θ)
    L, b = begin
        @unpack L, b = ComponentArray(θ, proto_axes)
        LowerTriangular(L), b
    end
    # For this to represent a covariance matrix we need to ensure that the diagonal is positive.
    # We can enforce this by zeroing out the diagonal and then adding back the diagonal exponentiated.
    D = Diagonal(diag(L))
    A = L - D + exp(D) # exp for Diagonal is the same as exponentiating only the diagonal entries

    b = to_constrained ∘ Shift(b) ∘ Scale(A)

    return transformed(base_dist, b)
end
```

```
getq (generic function with 1 method)
```



```julia
advi = ADVI(10, 20_000)
```

```
AdvancedVI.ADVI{AdvancedVI.ForwardDiffAD{0}}(10, 20000)
```



```julia
q_full_normal = vi(
    m, advi, getq, randn(num_params); optimizer=Variational.DecayedADAGrad(1e-2)
);
```




Let's have a look at the learned covariance matrix:

```julia
A = q_full_normal.transform.inner.a
```

```
13×13 LinearAlgebra.LowerTriangular{Float64, Matrix{Float64}}:
  0.317743      ⋅           …    ⋅           ⋅           ⋅ 
 -0.0029663    0.0165384         ⋅           ⋅           ⋅ 
 -0.0051169   -0.00180893        ⋅           ⋅           ⋅ 
  0.00343124   0.000239442       ⋅           ⋅           ⋅ 
  0.00517496  -0.00571897        ⋅           ⋅           ⋅ 
  0.00191448  -0.00123951   …    ⋅           ⋅           ⋅ 
  0.00288242   0.00798558        ⋅           ⋅           ⋅ 
  0.00240556   0.00653815        ⋅           ⋅           ⋅ 
 -0.00728202  -0.000471385       ⋅           ⋅           ⋅ 
 -0.00426615   0.000968262       ⋅           ⋅           ⋅ 
 -0.00221938   0.00148355   …   0.0235466    ⋅           ⋅ 
  0.0142979   -0.013414        -0.0198216   0.0173049    ⋅ 
 -0.0192627   -0.000503378      0.00284103  0.00188512  0.0164432
```



```julia
heatmap(cov(A * A'))
```

![](figures/09_variational-inference_68_1.png)

```julia
zs = rand(q_full_normal, 10_000);
```


```julia
p1 = plot_variational_marginals(rand(q_mf_normal, 10_000), sym2range)
p2 = plot_variational_marginals(rand(q_full_normal, 10_000), sym2range)

plot(p1, p2; layout=(1, 2), size=(800, 2000))
```

![](figures/09_variational-inference_70_1.png)



So it seems like the "full" ADVI approach, i.e. no mean-field assumption, obtain the same modes as the mean-field approach but with greater uncertainty for some of the `coefficients`. This

```julia
# Unfortunately, it seems like this has quite a high variance which is likely to be due to numerical instability, 
# so we consider a larger number of samples. If we get a couple of outliers due to numerical issues, 
# these kind affect the mean prediction greatly.
z = rand(q_full_normal, 10_000);
```


```julia
train_cut.VIFullPredictions = unstandardize(
    prediction(z, sym2range, train), train_unstandardized.MPG
)
test_cut.VIFullPredictions = unstandardize(
    prediction(z, sym2range, test), train_unstandardized.MPG
);
```


```julia
vi_loss1 = mean((train_cut.VIPredictions - train_cut.MPG) .^ 2)
vifull_loss1 = mean((train_cut.VIFullPredictions - train_cut.MPG) .^ 2)
bayes_loss1 = mean((train_cut.BayesPredictions - train_cut.MPG) .^ 2)
ols_loss1 = mean((train_cut.OLSPrediction - train_cut.MPG) .^ 2)

vi_loss2 = mean((test_cut.VIPredictions - test_cut.MPG) .^ 2)
vifull_loss2 = mean((test_cut.VIFullPredictions - test_cut.MPG) .^ 2)
bayes_loss2 = mean((test_cut.BayesPredictions - test_cut.MPG) .^ 2)
ols_loss2 = mean((test_cut.OLSPrediction - test_cut.MPG) .^ 2)

println("Training set:
    VI loss: $vi_loss1
    Bayes loss: $bayes_loss1
    OLS loss: $ols_loss1
Test set: 
    VI loss: $vi_loss2
    Bayes loss: $bayes_loss2
    OLS loss: $ols_loss2")
```

```
Training set:
    VI loss: 0.0013757680639403039
    Bayes loss: 9.181568032817367e-14
    OLS loss: 3.0709261248930093
Test set: 
    VI loss: 0.0014199510514292773
    Bayes loss: 1.3432106952596584e-12
    OLS loss: 27.09481307076057
```




```julia
z = rand(q_mf_normal, 1000);
preds = mapreduce(hcat, eachcol(z)) do zi
    return unstandardize(prediction(zi, sym2range, test), train_unstandardized.MPG)
end

p1 = scatter(
    1:size(test, 1),
    mean(preds; dims=2);
    yerr=std(preds; dims=2),
    label="prediction (mean ± std)",
    size=(900, 500),
    markersize=8,
)
scatter!(1:size(test, 1), unstandardize(test_label, train_unstandardized.MPG); label="true")
xaxis!(1:size(test, 1))
ylims!(10, 40)
title!("Mean-field ADVI (Normal)")
```

![](figures/09_variational-inference_75_1.png)

```julia
z = rand(q_full_normal, 1000);
preds = mapreduce(hcat, eachcol(z)) do zi
    return unstandardize(prediction(zi, sym2range, test), train_unstandardized.MPG)
end

p2 = scatter(
    1:size(test, 1),
    mean(preds; dims=2);
    yerr=std(preds; dims=2),
    label="prediction (mean ± std)",
    size=(900, 500),
    markersize=8,
)
scatter!(1:size(test, 1), unstandardize(test_label, train_unstandardized.MPG); label="true")
xaxis!(1:size(test, 1))
ylims!(10, 40)
title!("Full ADVI (Normal)")
```

![](figures/09_variational-inference_76_1.png)

```julia
preds = mapreduce(hcat, 1:5:size(chain, 1)) do i
    return unstandardize(prediction_chain(chain[i], test), train_unstandardized.MPG)
end

p3 = scatter(
    1:size(test, 1),
    mean(preds; dims=2);
    yerr=std(preds; dims=2),
    label="prediction (mean ± std)",
    size=(900, 500),
    markersize=8,
)
scatter!(1:size(test, 1), unstandardize(test_label, train_unstandardized.MPG); label="true")
xaxis!(1:size(test, 1))
ylims!(10, 40)
title!("MCMC (NUTS)")
```

![](figures/09_variational-inference_77_1.png)

```julia
plot(p1, p2, p3; layout=(1, 3), size=(900, 250), label="")
```

![](figures/09_variational-inference_78_1.png)



Here we actually see that indeed both the full ADVI and the MCMC approaches does a much better job of quantifying the uncertainty of predictions for never-before-seen samples, with full ADVI seemingly *underestimating* the variance slightly compared to MCMC.

So now you know how to do perform VI on your Turing.jl model! Great isn't it?


## Appendix

These tutorials are a part of the TuringTutorials repository, found at: [https://github.com/TuringLang/TuringTutorials](https://github.com/TuringLang/TuringTutorials).

To locally run this tutorial, do the following commands:

```
using TuringTutorials
TuringTutorials.weave("09-variational-inference", "09_variational-inference.jmd")
```

Computer Information:

```
Julia Version 1.9.2
Commit e4ee485e909 (2023-07-05 09:39 UTC)
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver2)
  Threads: 1 on 16 virtual cores
Environment:
  JULIA_CPU_THREADS = 16
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/7aa0085e-79a4-45f3-a5bd-9743c91cf3da
  JULIA_IMAGE_THREADS = 1

```

Package Information:

```
Status `/cache/build/default-amdci4-2/julialang/turingtutorials/tutorials/09-variational-inference/Project.toml`
⌅ [76274a88] Bijectors v0.12.8
  [b0b7db55] ComponentArrays v0.14.0
  [1a297f60] FillArrays v1.5.0
  [587475ba] Flux v0.14.1
  [38e38edf] GLM v1.8.3
  [442fdcdd] Measures v0.3.2
  [ce6b1742] RDatasets v0.7.7
  [f3b207a7] StatsPlots v0.15.6
  [fce5fe82] Turing v0.27.0
  [3a884ed6] UnPack v1.0.2
  [37e2e46d] LinearAlgebra
  [9a3f8284] Random
Info Packages marked with ⌅ have new versions available but compatibility constraints restrict them from upgrading. To see why use `status --outdated`
```

And the full manifest:

```
Status `/cache/build/default-amdci4-2/julialang/turingtutorials/tutorials/09-variational-inference/Manifest.toml`
  [47edcb42] ADTypes v0.1.6
  [621f4979] AbstractFFTs v1.4.0
  [80f14c24] AbstractMCMC v4.4.2
⌅ [7a57a42e] AbstractPPL v0.5.4
  [1520ce14] AbstractTrees v0.4.4
  [79e6a3ab] Adapt v3.6.2
⌅ [0bf59076] AdvancedHMC v0.4.6
  [5b7e9947] AdvancedMH v0.7.5
  [576499cb] AdvancedPS v0.4.3
  [b5ca4192] AdvancedVI v0.2.4
  [dce04be8] ArgCheck v2.3.0
  [7d9fca2a] Arpack v0.5.4
  [4fba245c] ArrayInterface v7.4.11
  [a9b6321e] Atomix v0.1.0
  [13072b0f] AxisAlgorithms v1.0.1
  [39de3d68] AxisArrays v0.4.7
  [198e06fe] BangBang v0.3.39
  [9718e550] Baselet v0.1.1
⌅ [76274a88] Bijectors v0.12.8
  [d1d4a3ce] BitFlags v0.1.7
  [fa961155] CEnum v0.4.2
  [336ed68f] CSV v0.10.11
  [49dc2e85] Calculus v0.5.1
  [324d7699] CategoricalArrays v0.10.8
  [082447d4] ChainRules v1.53.0
  [d360d2e6] ChainRulesCore v1.16.0
  [9e997f8a] ChangesOfVariables v0.1.8
  [aaaa29a8] Clustering v0.15.4
  [944b1d66] CodecZlib v0.7.2
  [35d6a980] ColorSchemes v3.22.0
  [3da002f7] ColorTypes v0.11.4
  [c3611d14] ColorVectorSpace v0.10.0
  [5ae59095] Colors v0.12.10
  [861a8166] Combinatorics v1.0.2
  [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v4.8.0
  [b0b7db55] ComponentArrays v0.14.0
  [a33af91c] CompositionsBase v0.1.2
  [f0e56b4a] ConcurrentUtilities v2.2.1
  [88cd18e8] ConsoleProgressMonitor v0.1.2
  [187b0558] ConstructionBase v1.5.3
  [6add18c4] ContextVariablesX v0.1.3
  [d38c429a] Contour v0.6.2
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.15.0
  [a93c6f00] DataFrames v1.6.1
  [864edb3b] DataStructures v0.18.14
  [e2d170a0] DataValueInterfaces v1.0.0
  [244e2a9f] DefineSingletons v0.1.2
  [8bb1440f] DelimitedFiles v1.9.1
  [b429d917] DensityInterface v0.4.0
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
  [b4f34e82] Distances v0.10.9
  [31c24e10] Distributions v0.25.98
  [ced4e74d] DistributionsAD v0.6.51
  [ffbed154] DocStringExtensions v0.9.3
  [fa6b7ba4] DualNumbers v0.6.8
⌃ [366bfd00] DynamicPPL v0.23.0
  [cad2338a] EllipticalSliceSampling v1.1.0
  [4e289a0a] EnumX v1.0.4
  [460bff9d] ExceptionUnwrapping v0.1.9
  [e2ba6199] ExprTools v0.1.10
  [c87230d0] FFMPEG v0.4.1
  [7a1cc6ca] FFTW v1.7.1
  [cc61a311] FLoops v0.2.1
  [b9860ae5] FLoopsBase v0.1.1
  [5789e2e9] FileIO v1.16.1
  [48062228] FilePathsBase v0.9.20
  [1a297f60] FillArrays v1.5.0
  [53c48c17] FixedPointNumbers v0.8.4
  [587475ba] Flux v0.14.1
  [59287772] Formatting v0.4.2
  [f6369f11] ForwardDiff v0.10.35
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
  [d9f16b24] Functors v0.4.5
  [38e38edf] GLM v1.8.3
  [0c68f7d7] GPUArrays v8.8.1
  [46192b85] GPUArraysCore v0.1.5
  [28b8d3ca] GR v0.72.9
  [42e2da0e] Grisu v1.0.2
  [cd3eb016] HTTP v1.9.14
  [34004b35] HypergeometricFunctions v0.3.23
  [7869d1d1] IRTools v0.4.10
  [615f187c] IfElse v0.1.1
  [22cec73e] InitialValues v0.3.1
  [842dd82b] InlineStrings v1.4.0
  [505f98c9] InplaceOps v0.3.0
  [a98d9a8b] Interpolations v0.14.7
  [8197267c] IntervalSets v0.7.5
  [3587e190] InverseFunctions v0.1.11
  [41ab1584] InvertedIndices v1.3.0
  [92d709cd] IrrationalConstants v0.2.2
  [c8e1da08] IterTools v1.8.0
  [82899510] IteratorInterfaceExtensions v1.0.0
  [1019f520] JLFzf v0.1.5
  [692b3bcd] JLLWrappers v1.4.1
  [682c06a0] JSON v0.21.4
  [b14d175d] JuliaVariables v0.2.4
  [63c18a36] KernelAbstractions v0.9.8
  [5ab0869b] KernelDensity v0.6.7
  [929cbde3] LLVM v6.1.0
  [8ac3fa9e] LRUCache v1.4.1
  [b964fa9f] LaTeXStrings v1.3.0
  [23fbe1c1] Latexify v0.16.1
  [50d2b5c4] Lazy v0.15.1
  [1d6d02ad] LeftChildRightSiblingTrees v0.2.0
  [6f1fad26] Libtask v0.8.6
  [6fdf6af0] LogDensityProblems v2.1.1
  [996a588d] LogDensityProblemsAD v1.6.1
  [2ab3a3ac] LogExpFunctions v0.3.24
  [e6f89c97] LoggingExtras v1.0.0
  [c7f686f2] MCMCChains v6.0.3
  [be115224] MCMCDiagnosticTools v0.3.4
  [e80e1ace] MLJModelInterface v1.8.0
  [d8e11817] MLStyle v0.4.17
  [f1d291b0] MLUtils v0.4.3
  [1914dd2f] MacroTools v0.5.10
  [dbb5928d] MappedArrays v0.4.2
  [739be429] MbedTLS v1.1.7
  [442fdcdd] Measures v0.3.2
  [128add7d] MicroCollections v0.1.4
  [e1d29d7a] Missings v1.1.0
  [78c3b35d] Mocking v0.7.7
  [6f286f6a] MultivariateStats v0.10.2
  [872c559c] NNlib v0.9.4
  [77ba4419] NaNMath v1.0.2
  [71a1bf82] NameResolution v0.1.5
  [86f7a689] NamedArrays v0.9.8
  [c020b1a1] NaturalSort v1.0.0
  [b8a86587] NearestNeighbors v0.4.13
  [510215fc] Observables v0.5.4
  [6fe1bfb0] OffsetArrays v1.12.10
  [0b1bfda6] OneHotArrays v0.2.4
  [4d8831e6] OpenSSL v1.4.1
  [3bd65402] Optimisers v0.2.19
  [bac558e1] OrderedCollections v1.6.2
  [90014a1f] PDMats v0.11.17
  [69de0a69] Parsers v2.7.1
  [b98c9c47] Pipe v1.3.0
  [ccf2f8ad] PlotThemes v3.1.0
  [995b91a9] PlotUtils v1.3.5
  [91a5bcdd] Plots v1.38.17
  [2dfb63ee] PooledArrays v1.4.2
  [aea7be01] PrecompileTools v1.1.2
  [21216c6a] Preferences v1.4.0
  [8162dcfd] PrettyPrint v0.2.0
  [08abe8d2] PrettyTables v2.2.7
  [33c8b6b6] ProgressLogging v0.1.4
  [92933f4c] ProgressMeter v1.7.2
  [1fd47b50] QuadGK v2.8.2
⌅ [df47a6cb] RData v0.8.3
  [ce6b1742] RDatasets v0.7.7
  [74087812] Random123 v1.6.1
  [e6cf234a] RandomNumbers v1.5.3
  [b3c3ace0] RangeArrays v0.3.2
  [c84ed2f1] Ratios v0.4.5
  [c1ae055f] RealDot v0.1.0
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
  [731186ca] RecursiveArrayTools v2.38.7
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.0
  [ae029012] Requires v1.3.0
  [79098fc4] Rmath v0.7.1
  [f2b01f46] Roots v2.0.17
  [7e49a35a] RuntimeGeneratedFunctions v0.5.11
  [0bca4576] SciMLBase v1.94.0
  [c0aeaf25] SciMLOperators v0.3.5
  [30f210dd] ScientificTypesBase v3.0.0
  [6c6a2e73] Scratch v1.2.0
  [91c51154] SentinelArrays v1.4.0
  [efcf1570] Setfield v1.1.1
  [1277b4bf] ShiftedArrays v2.0.0
  [605ecd9f] ShowCases v0.1.0
  [992d4aef] Showoff v1.0.3
  [777ac1f9] SimpleBufferStream v1.1.0
  [699a6c99] SimpleTraits v0.9.4
  [ce78b400] SimpleUnPack v1.1.0
  [66db9d55] SnoopPrecompile v1.0.3
  [a2af1166] SortingAlgorithms v1.1.1
  [276daf66] SpecialFunctions v2.3.0
  [171d559e] SplittablesBase v0.1.15
  [aedffcd0] Static v0.8.8
  [0d7ed370] StaticArrayInterface v1.4.0
  [90137ffa] StaticArrays v1.6.2
  [1e83bf80] StaticArraysCore v1.4.2
  [64bff920] StatisticalTraits v3.2.0
  [82ae8749] StatsAPI v1.6.0
  [2913bbd2] StatsBase v0.34.0
  [4c63d2b9] StatsFuns v1.3.0
  [3eaba693] StatsModels v0.7.2
  [f3b207a7] StatsPlots v0.15.6
  [892a3eda] StringManipulation v0.3.0
  [09ab397b] StructArrays v0.6.15
  [2efcf032] SymbolicIndexingInterface v0.2.2
  [ab02a1b2] TableOperations v1.2.0
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.10.1
  [62fd8b95] TensorCore v0.1.1
  [5d786b92] TerminalLoggers v0.1.7
  [f269a46b] TimeZones v1.11.0
  [9f7883ad] Tracker v0.2.26
  [3bb67fe8] TranscodingStreams v0.9.13
  [28d57a85] Transducers v0.4.78
  [410a4b4d] Tricks v0.1.7
  [781d530d] TruncatedStacktraces v1.4.0
  [fce5fe82] Turing v0.27.0
  [5c2747f8] URIs v1.4.2
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
  [1986cc42] Unitful v1.15.0
  [45397f5d] UnitfulLatexify v1.6.3
  [013be700] UnsafeAtomics v0.2.1
  [d80eeb9a] UnsafeAtomicsLLVM v0.1.3
  [41fe7b60] Unzip v0.2.0
  [ea10d353] WeakRefStrings v1.4.2
  [cc8bc4a8] Widgets v0.6.6
  [efce3f68] WoodburyMatrices v0.5.5
  [76eceee3] WorkerUtilities v1.6.1
  [e88e6eb3] Zygote v0.6.62
  [700de1a5] ZygoteRules v0.2.3
⌅ [68821587] Arpack_jll v3.5.1+1
  [6e34b625] Bzip2_jll v1.0.8+0
  [83423d85] Cairo_jll v1.16.1+1
  [2e619515] Expat_jll v2.5.0+0
  [b22a6f82] FFMPEG_jll v4.4.2+2
  [f5851436] FFTW_jll v3.3.10+0
  [a3f928ae] Fontconfig_jll v2.13.93+0
  [d7e528f0] FreeType2_jll v2.13.1+0
  [559328eb] FriBidi_jll v1.0.10+0
  [0656b61e] GLFW_jll v3.3.8+0
  [d2c73de3] GR_jll v0.72.9+0
  [78b55507] Gettext_jll v0.21.0+0
  [7746bdde] Glib_jll v2.74.0+2
  [3b182d85] Graphite2_jll v1.3.14+0
  [2e76f6c2] HarfBuzz_jll v2.8.1+1
  [1d5cc7b8] IntelOpenMP_jll v2023.1.0+0
  [aacddb02] JpegTurbo_jll v2.1.91+0
  [c1c5ebd0] LAME_jll v3.100.1+0
  [88015f11] LERC_jll v3.0.0+1
  [dad2f222] LLVMExtra_jll v0.0.23+0
  [1d63c593] LLVMOpenMP_jll v15.0.4+0
  [dd4b983a] LZO_jll v2.10.1+0
⌅ [e9f186c6] Libffi_jll v3.2.2+1
  [d4300ac3] Libgcrypt_jll v1.8.7+0
  [7e76a0d4] Libglvnd_jll v1.6.0+0
  [7add5ba3] Libgpg_error_jll v1.42.0+0
  [94ce4f54] Libiconv_jll v1.16.1+2
  [4b2f31a3] Libmount_jll v2.35.0+0
  [89763e89] Libtiff_jll v4.5.1+1
  [38a345b3] Libuuid_jll v2.36.0+0
  [856f044c] MKL_jll v2023.1.0+0
  [e7412a2a] Ogg_jll v1.3.5+1
⌅ [458c3c95] OpenSSL_jll v1.1.21+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.2+0
  [30392449] Pixman_jll v0.42.2+0
  [c0090381] Qt6Base_jll v6.4.2+3
  [f50d1b31] Rmath_jll v0.4.0+0
  [a2964d1f] Wayland_jll v1.21.0+0
  [2381bf8a] Wayland_protocols_jll v1.25.0+0
  [02c8fc9c] XML2_jll v2.10.3+0
  [aed1982a] XSLT_jll v1.1.34+0
  [ffd25f8a] XZ_jll v5.4.3+1
  [4f6342f7] Xorg_libX11_jll v1.8.6+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.11+0
  [935fb764] Xorg_libXcursor_jll v1.2.0+4
  [a3789734] Xorg_libXdmcp_jll v1.1.4+0
  [1082639a] Xorg_libXext_jll v1.3.4+4
  [d091e8ba] Xorg_libXfixes_jll v5.0.3+4
  [a51aa0fd] Xorg_libXi_jll v1.7.10+4
  [d1454406] Xorg_libXinerama_jll v1.1.4+4
  [ec84b674] Xorg_libXrandr_jll v1.5.2+4
  [ea2f1a96] Xorg_libXrender_jll v0.9.10+4
  [14d82f49] Xorg_libpthread_stubs_jll v0.1.1+0
  [c7cfdc94] Xorg_libxcb_jll v1.15.0+0
  [cc61e674] Xorg_libxkbfile_jll v1.1.2+0
  [12413925] Xorg_xcb_util_image_jll v0.4.0+1
  [2def613f] Xorg_xcb_util_jll v0.4.0+1
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.0+1
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.9+1
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.1+1
  [35661453] Xorg_xkbcomp_jll v1.4.6+0
  [33bec58e] Xorg_xkeyboard_config_jll v2.39.0+0
  [c5fb5394] Xorg_xtrans_jll v1.5.0+0
  [3161d3a3] Zstd_jll v1.5.5+0
⌅ [214eeab7] fzf_jll v0.29.0+0
  [a4ae2306] libaom_jll v3.4.0+0
  [0ac62f75] libass_jll v0.15.1+0
  [f638f0a6] libfdk_aac_jll v2.0.2+0
  [b53b4c65] libpng_jll v1.6.38+0
  [f27f6e37] libvorbis_jll v1.3.7+1
  [1270edf5] x264_jll v2021.5.5+0
  [dfaa095f] x265_jll v3.5.0+0
  [d8fb68d0] xkbcommon_jll v1.4.1+0
  [0dad84c5] ArgTools v1.1.1
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [ade2ca70] Dates
  [8ba89e20] Distributed
  [f43a241f] Downloads v1.6.0
  [7b1f6079] FileWatching
  [9fa8497b] Future
  [b77e0a4c] InteractiveUtils
  [4af54fe1] LazyArtifacts
  [b27032c2] LibCURL v0.6.3
  [76f85450] LibGit2
  [8f399da3] Libdl
  [37e2e46d] LinearAlgebra
  [56ddb016] Logging
  [d6f4376e] Markdown
  [a63ad114] Mmap
  [ca575930] NetworkOptions v1.2.0
  [44cfe95a] Pkg v1.9.2
  [de0858da] Printf
  [3fa0cd96] REPL
  [9a3f8284] Random
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization
  [1a1011a3] SharedArrays
  [6462fe0b] Sockets
  [2f01184e] SparseArrays
  [10745b16] Statistics v1.9.0
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll v1.0.5+0
  [deac9b47] LibCURL_jll v7.84.0+0
  [29816b5a] LibSSH2_jll v1.10.2+0
  [c8ffd9c3] MbedTLS_jll v2.28.2+0
  [14a3606d] MozillaCACerts_jll v2022.10.11
  [4536629a] OpenBLAS_jll v0.3.21+4
  [05823500] OpenLibm_jll v0.8.1+0
  [efcefdf7] PCRE2_jll v10.42.0+0
  [bea87d4a] SuiteSparse_jll v5.10.1+6
  [83775a58] Zlib_jll v1.2.13+0
  [8e850b90] libblastrampoline_jll v5.8.0+0
  [8e850ede] nghttp2_jll v1.48.0+0
  [3f19e933] p7zip_jll v17.4.0+0
Info Packages marked with ⌃ and ⌅ have new versions available, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
```

