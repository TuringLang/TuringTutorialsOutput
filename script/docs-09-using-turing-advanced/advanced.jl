
using Distributions, Turing, Random, Bijectors


struct CustomUniform <: ContinuousUnivariateDistribution end


# sample in [0, 1]
Distributions.rand(rng::AbstractRNG, d::CustomUniform) = rand(rng)

# p(x) = 1 → logp(x) = 0
Distributions.logpdf(d::CustomUniform, x::Real) = zero(x)


Bijectors.bijector(d::CustomUniform) = Logit(0.0, 1.0)


Distributions.minimum(d::CustomUniform) = 0.0
Distributions.maximum(d::CustomUniform) = 1.0


dist = Gamma(2,3)
b = bijector(dist)
transformed_dist = transformed(dist, b) # results in distribution with transformed support + correction for logpdf


using Turing

myloglikelihood(x, μ) = loglikelihood(Normal(μ, 1), x)

@model function demo(x)
    μ ~ Normal()
    Turing.@addlogprob! myloglikelihood(x, μ)
end


using Turing
using LinearAlgebra

@model function demo(x)
    m ~ MvNormal(zero(x), I)
    if dot(m, x) < 0
        Turing.@addlogprob! -Inf
        # Exit the model evaluation early
        return nothing
    end

    x ~ MvNormal(m, I)
    return nothing
end


if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
    Turing.@addlogprob! myloglikelihood(x, μ)
end


using Turing

@model function gdemo(x)
    # Set priors.
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))

    # Observe each value of x.
    @. x ~ Normal(m, sqrt(s²))
end

model = gdemo([1.5, 2.0])


using Turing

# Create the model function.
function gdemo(model, varinfo, context, x)
    # Assume s² has an InverseGamma distribution.
    s², varinfo = DynamicPPL.tilde_assume!!(
        context, InverseGamma(2, 3), Turing.@varname(s²), varinfo
    )

    # Assume m has a Normal distribution.
    m, varinfo = DynamicPPL.tilde_assume!!(
        context, Normal(0, sqrt(s²)), Turing.@varname(m), varinfo
    )

    # Observe each value of x[i] according to a Normal distribution.
    return DynamicPPL.dot_tilde_observe!!(
        context, Normal(m, sqrt(s²)), x, Turing.@varname(x), varinfo
    )
end
gdemo(x) = Turing.Model(gdemo, (; x))

# Instantiate a Model object with our data variables.
model = gdemo([1.5, 2.0])

