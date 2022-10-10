
using Turing
using AbstractGPs
using DataFrames
using FillArrays
using RDatasets
using StatsBase
using StatsPlots
using VegaLite

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


@model function pPCA(x, ::Type{TV}=Array{Float64}) where {TV}
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
end;


linear_kernel(α) = LinearKernel() ∘ ARDTransform(α)
sekernel(α, σ) = σ * SqExponentialKernel() ∘ ARDTransform(α);


@model function GPLVM_linear(Y, K=4, ::Type{T}=Float64) where {T}

    # Dimensionality of the problem.
    N, D = size(Y)
    # K is the dimension of the latent space
    @assert K <= D
    noise = 1e-3

    # Priors
    α ~ MvLogNormal(MvNormal(Zeros(K), I))
    Z ~ filldist(Normal(), K, N)
    mu ~ filldist(Normal(), N)

    kernel = linear_kernel(α)

    gp = GP(mu, kernel)
    cv = cov(gp(ColVecs(Z), noise))
    return Y ~ filldist(MvNormal(mu, cv), D)
end;

@model function GPLVM(Y, K=4, ::Type{T}=Float64) where {T}

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

    kernel = sekernel(α, σ)

    gp = GP(mu, kernel)
    cv = cov(gp(ColVecs(Z), noise))
    return Y ~ filldist(MvNormal(mu, cv), D)
end;


# Standard GPs don't scale very well in n, so we use a small subsample for the purpose of this tutorial
n_data = 40
# number of features to use from dataset
n_features = 4
# latent dimension for GP case
ndim = 4;


ppca = pPCA(dat[1:n_data, 1:n_features])
chain_ppca = sample(ppca, NUTS(), 1000);


# we extract the posterior mean estimates of the parameters from the chain
w = reshape(mean(group(chain_ppca, :w))[:, 2], (n_features, n_features))
z = reshape(mean(group(chain_ppca, :z))[:, 2], (n_features, n_data))
X = w * z

df_pre = DataFrame(z', :auto)
rename!(df_pre, Symbol.(["z" * string(i) for i in collect(1:n_features)]))
df_pre[!, :type] = labels[1:n_data]
p_ppca = @vlplot(:point, x = :z1, y = :z2, color = "type:n")(df_pre)


gplvm_linear = GPLVM_linear(dat[1:n_data, 1:n_features], ndim)

chain_linear = sample(gplvm_linear, NUTS(), 500)
# we extract the posterior mean estimates of the parameters from the chain
z_mean = reshape(mean(group(chain_linear, :Z))[:, 2], (n_features, n_data))
alpha_mean = mean(group(chain_linear, :α))[:, 2]


df_gplvm_linear = DataFrame(z_mean', :auto)
rename!(df_gplvm_linear, Symbol.(["z" * string(i) for i in collect(1:ndim)]))
df_gplvm_linear[!, :sample] = 1:n_data
df_gplvm_linear[!, :labels] = labels[1:n_data]
alpha_indices = sortperm(alpha_mean; rev=true)[1:2]
println(alpha_indices)
df_gplvm_linear[!, :ard1] = z_mean[alpha_indices[1], :]
df_gplvm_linear[!, :ard2] = z_mean[alpha_indices[2], :]

p_linear = @vlplot(:point, x = :ard1, y = :ard2, color = "labels:n")(df_gplvm_linear)
p_linear


gplvm = GPLVM(dat[1:n_data, 1:n_features], ndim)

chain_gplvm = sample(gplvm, NUTS(), 500)
# we extract the posterior mean estimates of the parameters from the chain
z_mean = reshape(mean(group(chain_gplvm, :Z))[:, 2], (ndim, n_data))
alpha_mean = mean(group(chain_gplvm, :α))[:, 2]


df_gplvm = DataFrame(z_mean', :auto)
rename!(df_gplvm, Symbol.(["z" * string(i) for i in collect(1:ndim)]))
df_gplvm[!, :sample] = 1:n_data
df_gplvm[!, :labels] = labels[1:n_data]
alpha_indices = sortperm(alpha_mean; rev=true)[1:2]
println(alpha_indices)
df_gplvm[!, :ard1] = z_mean[alpha_indices[1], :]
df_gplvm[!, :ard2] = z_mean[alpha_indices[2], :]

p_gplvm = @vlplot(:point, x = :ard1, y = :ard2, color = "labels:n")(df_gplvm)
p_gplvm


let
    @assert abs(
        mean(z_mean[alpha_indices[1], labels[1:n_data] .== "setosa"]) -
        mean(z_mean[alpha_indices[1], labels[1:n_data] .!= "setosa"]),
    ) > 1.4
end


using Stheno
@model function GPLVM_sparse(Y, K, ::Type{T}=Float64) where {T}

    # Dimensionality of the problem.
    N, D = size(Y)
    # dimension of latent space
    @assert K <= D
    # number of inducing points
    n_inducing = 25
    noise = 1e-3

    # Priors
    α ~ MvLogNormal(MvNormal(Zeros(K), I))
    σ ~ LogNormal(1.0, 1.0)
    Z ~ filldist(Normal(), K, N)
    mu ~ filldist(Normal(), N)

    kernel = σ * SqExponentialKernel() ∘ ARDTransform(α)

    ## Standard
    # gpc = GPC()
    # f = atomic(GP(kernel), gpc)
    # gp = f(ColVecs(Z), noise)
    # Y ~ filldist(gp, D)

    ## SPARSE GP
    #  xu = reshape(repeat(locations, K), :, K) # inducing points
    #  xu = reshape(repeat(collect(range(-2.0, 2.0; length=20)), K), :, K) # inducing points
    lbound = minimum(Y) + 1e-6
    ubound = maximum(Y) - 1e-6
    #  locations ~ filldist(Uniform(lbound, ubound), n_inducing)
    #  locations = [-2., -1.5 -1., -0.5, -0.25, 0.25, 0.5, 1., 2.]
    #  locations = collect(LinRange(lbound, ubound, n_inducing))
    locations = quantile(vec(Y), LinRange(0.01, 0.99, n_inducing))
    xu = reshape(locations, 1, :)
    gp = atomic(GP(kernel), GPC())
    fobs = gp(ColVecs(Z), noise)
    finducing = gp(xu, 1e-12)
    sfgp = SparseFiniteGP(fobs, finducing)
    cv = cov(sfgp.fobs)
    return Y ~ filldist(MvNormal(mu, cv), D)
end


n_data = 50
gplvm_sparse = GPLVM_sparse(dat[1:n_data, :], ndim)

chain_gplvm_sparse = sample(gplvm_sparse, NUTS(), 500)
# we extract the posterior mean estimates of the parameters from the chain
z_mean = reshape(mean(group(chain_gplvm_sparse, :Z))[:, 2], (ndim, n_data))
alpha_mean = mean(group(chain_gplvm_sparse, :α))[:, 2]


df_gplvm_sparse = DataFrame(z_mean', :auto)
rename!(df_gplvm_sparse, Symbol.(["z" * string(i) for i in collect(1:ndim)]))
df_gplvm_sparse[!, :sample] = 1:n_data
df_gplvm_sparse[!, :labels] = labels[1:n_data]
alpha_indices = sortperm(alpha_mean; rev=true)[1:2]
df_gplvm_sparse[!, :ard1] = z_mean[alpha_indices[1], :]
df_gplvm_sparse[!, :ard2] = z_mean[alpha_indices[2], :]
p_sparse = @vlplot(:point, x = :ard1, y = :ard2, color = "labels:n")(df_gplvm_sparse)
p_sparse

