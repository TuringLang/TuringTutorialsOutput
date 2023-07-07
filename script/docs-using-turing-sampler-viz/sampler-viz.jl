
ENV["GKS_ENCODING"] = "utf-8" # Allows the use of unicode characters in Plots.jl
using Plots
using StatsPlots
using Turing
using Random
using Bijectors

# Set a seed.
Random.seed!(0)

# Define a strange model.
@model function gdemo(x)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    bumps = sin(m) + cos(m)
    m = m + 5 * bumps
    for i in eachindex(x)
        x[i] ~ Normal(m, sqrt(s²))
    end
    return s², m
end

# Define our data points.
x = [1.5, 2.0, 13.0, 2.1, 0.0]

# Set up the model call, sample from the prior.
model = gdemo(x)

# Evaluate surface at coordinates.
evaluate(m1, m2) = logjoint(model, (m=m2, s²=invlink.(Ref(InverseGamma(2, 3)), m1)))

function plot_sampler(chain; label="")
    # Extract values from chain.
    val = get(chain, [:s², :m, :lp])
    ss = link.(Ref(InverseGamma(2, 3)), val.s²)
    ms = val.m
    lps = val.lp

    # How many surface points to sample.
    granularity = 100

    # Range start/stop points.
    spread = 0.5
    σ_start = minimum(ss) - spread * std(ss)
    σ_stop = maximum(ss) + spread * std(ss)
    μ_start = minimum(ms) - spread * std(ms)
    μ_stop = maximum(ms) + spread * std(ms)
    σ_rng = collect(range(σ_start; stop=σ_stop, length=granularity))
    μ_rng = collect(range(μ_start; stop=μ_stop, length=granularity))

    # Make surface plot.
    p = surface(
        σ_rng,
        μ_rng,
        evaluate;
        camera=(30, 65),
        #   ticks=nothing,
        colorbar=false,
        color=:inferno,
        title=label,
    )

    line_range = 1:length(ms)

    scatter3d!(
        ss[line_range],
        ms[line_range],
        lps[line_range];
        mc=:viridis,
        marker_z=collect(line_range),
        msw=0,
        legend=false,
        colorbar=false,
        alpha=0.5,
        xlabel="σ",
        ylabel="μ",
        zlabel="Log probability",
        title=label,
    )

    return p
end;


c = sample(model, Gibbs(HMC(0.01, 5, :s²), PG(20, :m)), 1000)
plot_sampler(c)


c = sample(model, HMC(0.01, 10), 1000)
plot_sampler(c)


c = sample(model, HMCDA(200, 0.65, 0.3), 1000)
plot_sampler(c)


c = sample(model, MH(), 1000)
plot_sampler(c)


c = sample(model, NUTS(0.65), 1000)
plot_sampler(c)


c = sample(model, NUTS(0.95), 1000)
plot_sampler(c)


c = sample(model, NUTS(0.2), 1000)
plot_sampler(c)


c = sample(model, PG(20), 1000)
plot_sampler(c)


c = sample(model, PG(50), 1000)
plot_sampler(c)

