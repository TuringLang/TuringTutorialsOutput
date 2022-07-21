
using StatsPlots, Turing, Statistics, LinearAlgebra, Random
Random.seed!(12345)
true_sin_freq = 2
true_sin_amp = 5
true_cos_freq = 7
true_cos_amp = 2.5
tmax = 10
β_true = 2
α_true = -1
tt = 0:0.05:tmax
f₁(t) = α_true + β_true * t
f₂(t) = true_sin_amp * sinpi(2 * t * true_sin_freq / tmax)
f₃(t) = true_cos_amp * cospi(2 * t * true_cos_freq / tmax)
f(t) = f₁(t) + f₂(t) + f₃(t)

plot(f, tt; label="f(t)", title="Observed time series", legend=:topleft, linewidth=3)
plot!(
    [f₁, f₂, f₃],
    tt;
    label=["f₁(t)" "f₂(t)" "f₃(t)"],
    style=[:dot :dash :dashdot],
    linewidth=1,
)


g(t) = f₁(t) * f₂(t) * f₃(t)

plot(g, tt; label="f(t)", title="Observed time series", legend=:topleft, linewidth=3)
plot!([f₁, f₂, f₃], tt; label=["f₁(t)" "f₂(t)" "f₃(t)"], linewidth=1)


σ_true = 0.35
t = collect(tt[begin:3:end])
t_min, t_max = extrema(t)
x = (t .- t_min) ./ (t_max - t_min)
yf = f.(t) .+ σ_true .* randn(size(t))
yf_max = maximum(yf)
yf = yf .- yf_max

scatter(x, yf; title="Standardised data", legend=false)


freqs = 1:10
num_freqs = length(freqs)
period = 1
cyclic_features = [sinpi.(2 .* freqs' .* x ./ period) cospi.(2 .* freqs' .* x ./ period)]

plot_freqs = [1, 3, 5]
freq_ptl = plot(
    cyclic_features[:, plot_freqs];
    label=permutedims(["sin(2π$(f)x)" for f in plot_freqs]),
    title="Cyclical features subset",
)


@model function decomp_model(t, c, op)
    α ~ Normal(0, 10)
    βt ~ Normal(0, 2)
    βc ~ MvNormal(zeros(size(c, 2)), I)
    σ ~ truncated(Normal(0, 0.1); lower=0)

    cyclic = c * βc
    trend = α .+ βt .* t
    μ = op(trend, cyclic)
    y ~ MvNormal(μ, σ^2 * I)
    return (; trend, cyclic)
end

y_prior_samples = mapreduce(hcat, 1:100) do _
    rand(decomp_model(t, cyclic_features, +)).y
end
plot(t, y_prior_samples; linewidth=1, alpha=0.5, color=1, label="", title="Prior samples")
scatter!(t, yf; color=2, label="Data")


function mean_ribbon(samples)
    qs = quantile(samples)
    low = qs[:, Symbol("2.5%")]
    up = qs[:, Symbol("97.5%")]
    m = mean(samples)[:, :mean]
    return m, (m - low, up - m)
end

function get_decomposition(model, x, cyclic_features, chain, op)
    chain_params = Turing.MCMCChains.get_sections(chain, :parameters)
    return generated_quantities(model(x, cyclic_features, op), chain_params)
end

function plot_fit(x, y, decomp, ymax)
    trend = mapreduce(x -> x.trend, hcat, decomp)
    cyclic = mapreduce(x -> x.cyclic, hcat, decomp)

    trend_plt = plot(
        x,
        trend .+ ymax;
        color=1,
        label=nothing,
        alpha=0.2,
        title="Trend",
        xlabel="Time",
        ylabel="f₁(t)",
    )
    ls = [ones(length(t)) t] \ y
    α̂, β̂ = ls[1], ls[2:end]
    plot!(
        trend_plt,
        t,
        α̂ .+ t .* β̂ .+ ymax;
        label="Least squares trend",
        color=5,
        linewidth=4,
    )

    scatter!(trend_plt, x, y .+ ymax; label=nothing, color=2, legend=:topleft)
    cyclic_plt = plot(
        x,
        cyclic;
        color=1,
        label=nothing,
        alpha=0.2,
        title="Cyclic effect",
        xlabel="Time",
        ylabel="f₂(t)",
    )
    return trend_plt, cyclic_plt
end

chain = sample(decomp_model(x, cyclic_features, +) | (; y=yf), NUTS(), 2000)
yf_samples = predict(decomp_model(x, cyclic_features, +), chain)
m, conf = mean_ribbon(yf_samples)
predictive_plt = plot(
    t,
    m .+ yf_max;
    ribbon=conf,
    label="Posterior density",
    title="Posterior decomposition",
    xlabel="Time",
    ylabel="f(t)",
)
scatter!(predictive_plt, t, yf .+ yf_max; color=2, label="Data", legend=:topleft)

decomp = get_decomposition(decomp_model, x, cyclic_features, chain, +)
decomposed_plt = plot_fit(t, yf, decomp, yf_max)
plot(predictive_plt, decomposed_plt...; layout=(3, 1), size=(700, 1000))


@assert isapprox(mean(chain, :βt) / t_max, β_true; atol=0.5)
@assert isapprox(mean(chain, :α) + yf_max, α_true; atol=1.0)
@assert isapprox(mean(chain, :σ), σ_true; atol=0.1)
@assert isapprox(mean(chain, "βc[2]"), true_sin_amp; atol=0.5)
@assert isapprox(mean(chain, "βc[17]"), true_cos_amp; atol=0.5)


function plot_cyclic_features(βsin, βcos)
    labels = reshape(["freq = $i" for i in freqs], 1, :)
    colors = collect(freqs)'
    style = reshape([i <= 10 ? :solid : :dash for i in 1:length(labels)], 1, :)
    sin_features_plt = density(
        βsin[:, :, 1];
        title="Sine features posterior",
        label=labels,
        ylabel="Density",
        xlabel="Weight",
        color=colors,
        linestyle=style,
        legend=nothing,
    )
    cos_features_plt = density(
        βcos[:, :, 1];
        title="Cosine features posterior",
        ylabel="Density",
        xlabel="Weight",
        label=nothing,
        color=colors,
        linestyle=style,
    )

    return seasonal_features_plt = plot(
        sin_features_plt,
        cos_features_plt;
        layout=(2, 1),
        size=(800, 600),
        legend=:outerright,
    )
end

βc = Array(group(chain, :βc))
plot_cyclic_features(βc[:, begin:num_freqs, :], βc[:, (num_freqs + 1):end, :])


yg = g.(t) .+ σ_true .* randn(size(t))

y_prior_samples = mapreduce(hcat, 1:100) do _
    rand(decomp_model(t, cyclic_features, .*)).y
end
plot(t, y_prior_samples; linewidth=1, alpha=0.5, color=1, label="", title="Prior samples")
scatter!(t, yf; color=2, label="Data")


chain = sample(decomp_model(x, cyclic_features, .*) | (; y=yg), NUTS(), 2000)
yg_samples = predict(decomp_model(x, cyclic_features, .*), chain)
m, conf = mean_ribbon(yg_samples)
predictive_plt = plot(
    t,
    m;
    ribbon=conf,
    label="Posterior density",
    title="Posterior decomposition",
    xlabel="Time",
    ylabel="g(t)",
)
scatter!(predictive_plt, t, yg; color=2, label="Data", legend=:topleft)

decomp = get_decomposition(decomp_model, x, cyclic_features, chain, .*)
decomposed_plt = plot_fit(t, yg, decomp, 0)
plot(predictive_plt, decomposed_plt...; layout=(3, 1), size=(700, 1000))


βc = Array(group(chain, :βc))
plot_cyclic_features(βc[:, begin:num_freqs, :], βc[:, (num_freqs + 1):end, :])


if isdefined(Main, :TuringTutorials)
    Main.TuringTutorials.tutorial_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])
end

