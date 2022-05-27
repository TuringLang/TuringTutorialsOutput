
using Distributions

using Random
Random.seed!(12); # Set seed for reproducibility


using StatsPlots


p_true = 0.5;


N = 100;


data = rand(Bernoulli(p_true), N);


data[1:5]


prior_belief = Beta(1, 1);


function updated_belief(prior_belief::Beta, data::AbstractArray{Bool})
    # Count the number of heads and tails.
    heads = sum(data)
    tails = length(data) - heads

    # Update our prior belief in closed form (this is possible because we use a conjugate prior).
    return Beta(prior_belief.α + heads, prior_belief.β + tails)
end

# Show updated belief for increasing number of observations
@gif for n in 0:N
    plot(
        updated_belief(prior_belief, data[1:n]);
        size=(500, 250),
        title="Updated belief after $n observations",
        xlabel="probability of heads",
        ylabel="",
        legend=nothing,
        xlim=(0, 1),
        fill=0,
        α=0.3,
        w=3,
    )
    vline!([p_true])
end


using Turing


using MCMCChains


# Unconditioned coinflip model with `N` observations.
@model function coinflip(; N::Int)
    # Our prior belief about the probability of heads in a coin toss.
    p ~ Beta(1, 1)

    # Heads or tails of a coin are drawn from `N` independent and identically
    # distributed Bernoulli distributions with success rate `p`.
    y ~ filldist(Bernoulli(p), N)

    return y
end

# Convenience method for constructing a coinflip model
# that is conditioned on observations `y`.
coinflip(y::AbstractVector{<:Real}) = coinflip(; N=length(y)) | (; y);


model = coinflip(data);


sampler = HMC(0.05, 10);


chain = sample(model, sampler, 1_000; progress=false);


histogram(chain)


@assert isapprox(mean(chain, :p), 0.5; atol=0.1)


# Visualize a blue density plot of the approximate posterior distribution using HMC (see Chain 1 in the legend).
density(chain; xlim=(0, 1), legend=:best, w=2, c=:blue)

# Visualize a green density plot of the posterior distribution in closed-form.
plot!(
    0:0.01:1,
    pdf.(updated_belief(prior_belief, data), 0:0.01:1);
    xlabel="probability of heads",
    ylabel="",
    title="",
    xlim=(0, 1),
    label="Closed-form",
    fill=0,
    α=0.3,
    w=3,
    c=:lightgreen,
)

# Visualize the true probability of heads in red.
vline!([p_true]; label="True probability", c=:red)

