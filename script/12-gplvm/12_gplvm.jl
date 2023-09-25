
using Turing
using AbstractGPs
using FillArrays
using LaTeXStrings
using Plots
using RDatasets
using ReverseDiff
using StatsBase

using LinearAlgebra
using Random

Random.seed!(1789);


data = dataset("datasets", "iris")
species = data[!, "Species"]
index = shuffle(1:150)
# we extract the four measured quantities,
# so the dimension of the data is only d=4 for this toy example
dat = Matrix(data[index, 1:4])
labels = data[index, "Species"]

# non-linearize data to demonstrate ability of GPs to deal with non-linearity
dat[:, 1] = 0.5 * dat[:, 1] .^ 2 + 0.1 * dat[:, 1] .^ 3
dat[:, 2] = dat[:, 2] .^ 3 + 0.2 * dat[:, 2] .^ 4
dat[:, 3] = 0.1 * exp.(dat[:, 3]) - 0.2 * dat[:, 3] .^ 2
dat[:, 4] = 0.5 * log.(dat[:, 4]) .^ 2 + 0.01 * dat[:, 3] .^ 5

# normalize data
dt = fit(ZScoreTransform, dat; dims=1);
StatsBase.transform!(dt, dat);


@model function pPCA(x)
    # Dimensionality of the problem.
    N, D = size(x)
    # latent variable z
    z ~ filldist(Normal(), D, N)
    # weights/loadings W
    w ~ filldist(Normal(), D, D)
    mu = (w * z)'
    for d in 1:D
        x[:, d] ~ MvNormal(mu[:, d], I)
    end
    return nothing
end;


linear_kernel(α) = LinearKernel() ∘ ARDTransform(α)
sekernel(α, σ) = σ * SqExponentialKernel() ∘ ARDTransform(α);


@model function GPLVM_linear(Y, K)
    # Dimensionality of the problem.
    N, D = size(Y)
    # K is the dimension of the latent space
    @assert K <= D
    noise = 1e-3

    # Priors
    α ~ MvLogNormal(MvNormal(Zeros(K), I))
    Z ~ filldist(Normal(), K, N)
    mu ~ filldist(Normal(), N)

    gp = GP(linear_kernel(α))
    gpz = gp(ColVecs(Z), noise)
    Y ~ filldist(MvNormal(mu, cov(gpz)), D)

    return nothing
end;

@model function GPLVM(Y, K)
    # Dimensionality of the problem.
    N, D = size(Y)
    # K is the dimension of the latent space
    @assert K <= D
    noise = 1e-3

    # Priors
    α ~ MvLogNormal(MvNormal(Zeros(K), I))
    σ ~ LogNormal(0.0, 1.0)
    Z ~ filldist(Normal(), K, N)
    mu ~ filldist(Normal(), N)

    gp = GP(sekernel(α, σ))
    gpz = gp(ColVecs(Z), noise)
    Y ~ filldist(MvNormal(mu, cov(gpz)), D)

    return nothing
end;


# Standard GPs don't scale very well in n, so we use a small subsample for the purpose of this tutorial
n_data = 40
# number of features to use from dataset
n_features = 4
# latent dimension for GP case
ndim = 4;


ppca = pPCA(dat[1:n_data, 1:n_features])
chain_ppca = sample(ppca, NUTS{Turing.ReverseDiffAD{true}}(), 1000);


# we extract the posterior mean estimates of the parameters from the chain
z_mean = reshape(mean(group(chain_ppca, :z))[:, 2], (n_features, n_data))
scatter(z_mean[1, :], z_mean[2, :]; group=labels[1:n_data], xlabel=L"z_1", ylabel=L"z_2")


gplvm_linear = GPLVM_linear(dat[1:n_data, 1:n_features], ndim)
chain_linear = sample(gplvm_linear, NUTS{Turing.ReverseDiffAD{true}}(), 500);


# we extract the posterior mean estimates of the parameters from the chain
z_mean = reshape(mean(group(chain_linear, :Z))[:, 2], (n_features, n_data))
alpha_mean = mean(group(chain_linear, :α))[:, 2]

alpha1, alpha2 = partialsortperm(alpha_mean, 1:2; rev=true)
scatter(
    z_mean[alpha1, :],
    z_mean[alpha2, :];
    group=labels[1:n_data],
    xlabel=L"z_{\mathrm{ard}_1}",
    ylabel=L"z_{\mathrm{ard}_2}",
)


gplvm = GPLVM(dat[1:n_data, 1:n_features], ndim)
chain_gplvm = sample(gplvm, NUTS{Turing.ReverseDiffAD{true}}(), 500);


# we extract the posterior mean estimates of the parameters from the chain
z_mean = reshape(mean(group(chain_gplvm, :Z))[:, 2], (ndim, n_data))
alpha_mean = mean(group(chain_gplvm, :α))[:, 2]

alpha1, alpha2 = partialsortperm(alpha_mean, 1:2; rev=true)
scatter(
    z_mean[alpha1, :],
    z_mean[alpha2, :];
    group=labels[1:n_data],
    xlabel=L"z_{\mathrm{ard}_1}",
    ylabel=L"z_{\mathrm{ard}_2}",
)


let
    @assert abs(
        mean(z_mean[alpha1, labels[1:n_data] .== "setosa"]) -
        mean(z_mean[alpha1, labels[1:n_data] .!= "setosa"]),
    ) > 1
end

