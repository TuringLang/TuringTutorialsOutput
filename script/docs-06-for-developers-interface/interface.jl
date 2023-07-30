
# Import the relevant libraries.
using AbstractMCMC: AbstractMCMC
using Distributions
using Random


# Define a sampler type.
struct MetropolisHastings{T,D} <: AbstractMCMC.AbstractSampler
    init_θ::T
    proposal::D
end

# Default constructors.
MetropolisHastings(init_θ::Real) = MetropolisHastings(init_θ, Normal(0, 1))
function MetropolisHastings(init_θ::Vector{<:Real})
    return MetropolisHastings(init_θ, MvNormal(zero(init_θ), I))
end


# Define a model type. Stores the log density function.
struct DensityModel{F<:Function} <: AbstractMCMC.AbstractModel
    ℓπ::F
end


# Create a very basic Transition type, only stores the 
# parameter draws and the log probability of the draw.
struct Transition{T,L}
    θ::T
    lp::L
end

# Store the new draw and its log density.
Transition(model::DensityModel, θ) = Transition(θ, ℓπ(model, θ))


# Define the first step! function, which is called at the 
# beginning of sampling. Return the initial parameter used
# to define the sampler.
function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::DensityModel,
    spl::MetropolisHastings,
    N::Integer,
    ::Nothing;
    kwargs...,
)
    return Transition(model, spl.init_θ)
end


# Define a function that makes a basic proposal depending on a univariate
# parameterization or a multivariate parameterization.
function propose(spl::MetropolisHastings, model::DensityModel, θ::Real)
    return Transition(model, θ + rand(spl.proposal))
end
function propose(spl::MetropolisHastings, model::DensityModel, θ::Vector{<:Real})
    return Transition(model, θ + rand(spl.proposal))
end
function propose(spl::MetropolisHastings, model::DensityModel, t::Transition)
    return propose(spl, model, t.θ)
end

# Calculates the probability `q(θ|θcond)`, using the proposal distribution `spl.proposal`.
q(spl::MetropolisHastings, θ::Real, θcond::Real) = logpdf(spl.proposal, θ - θcond)
function q(spl::MetropolisHastings, θ::Vector{<:Real}, θcond::Vector{<:Real})
    return logpdf(spl.proposal, θ - θcond)
end
q(spl::MetropolisHastings, t1::Transition, t2::Transition) = q(spl, t1.θ, t2.θ)

# Calculate the density of the model given some parameterization.
ℓπ(model::DensityModel, θ) = model.ℓπ(θ)
ℓπ(model::DensityModel, t::Transition) = t.lp

# Define the other step function. Returns a Transition containing
# either a new proposal (if accepted) or the previous proposal 
# (if not accepted).
function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::DensityModel,
    spl::MetropolisHastings,
    ::Integer,
    θ_prev::Transition;
    kwargs...,
)
    # Generate a new proposal.
    θ = propose(spl, model, θ_prev)

    # Calculate the log acceptance probability.
    α = ℓπ(model, θ) - ℓπ(model, θ_prev) + q(spl, θ_prev, θ) - q(spl, θ, θ_prev)

    # Decide whether to return the previous θ or the new one.
    if log(rand(rng)) < min(α, 0.0)
        return θ
    else
        return θ_prev
    end
end


# A basic chains constructor that works with the Transition struct we defined.
function AbstractMCMC.bundle_samples(
    rng::AbstractRNG,
    ℓ::DensityModel,
    s::MetropolisHastings,
    N::Integer,
    ts::Vector{<:Transition},
    chain_type::Type{Any};
    param_names=missing,
    kwargs...,
)
    # Turn all the transitions into a vector-of-vectors.
    vals = copy(reduce(hcat, [vcat(t.θ, t.lp) for t in ts])')

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = ["Parameter $i" for i in 1:(length(first(vals)) - 1)]
    end

    # Add the log density field to the parameter names.
    push!(param_names, "lp")

    # Bundle everything up and return a Chains struct.
    return Chains(vals, param_names, (internals=["lp"],))
end


# Generate a set of data from the posterior we want to estimate.
data = rand(Normal(5, 3), 30)

# Define the components of a basic model.
insupport(θ) = θ[2] >= 0
dist(θ) = Normal(θ[1], θ[2])
density(θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf

# Construct a DensityModel.
model = DensityModel(density)

# Set up our sampler with initial parameters.
spl = MetropolisHastings([0.0, 0.0])

# Sample from the posterior.
chain = sample(model, spl, 100000; param_names=["μ", "σ"])

