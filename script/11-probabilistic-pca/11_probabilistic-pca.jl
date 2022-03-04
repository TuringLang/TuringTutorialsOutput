
using Turing
using LinearAlgebra

# Packages for visualization
using VegaLite, DataFrames, StatsPlots

# Import Fisher's iris example data set
using RDatasets

# Set a seed for reproducibility.
using Random
Random.seed!(1789);


n_cells = 60
n_genes = 9
mu_1 = 10.0 * ones(n_genes ÷ 3)
mu_0 = zeros(n_genes ÷ 3)
S = I(n_genes ÷ 3)
mvn_0 = MvNormal(mu_0, S)
mvn_1 = MvNormal(mu_1, S)

# create a diagonal block like expression matrix, with some non-informative genes;
# not all features/genes are informative, some might just not differ very much between cells)
expression_matrix = transpose(
    vcat(
        hcat(rand(mvn_1, n_cells ÷ 2), rand(mvn_0, n_cells ÷ 2)),
        hcat(rand(mvn_0, n_cells ÷ 2), rand(mvn_0, n_cells ÷ 2)),
        hcat(rand(mvn_0, n_cells ÷ 2), rand(mvn_1, n_cells ÷ 2)),
    ),
)

df_exp = DataFrame(expression_matrix, :auto)
df_exp[!, :cell] = 1:n_cells

@vlplot(
    :rect,
    x = "cell:o",
    color = :value,
    encoding = {
        y = {field = "variable", type = "nominal", sort = "-x", axis = {title = "gene"}}
    }
)(
    DataFrames.stack(df_exp, 1:n_genes)
)


@model function pPCA(x, ::Type{TV}=Array{Float64}) where {TV}
    # Dimensionality of the problem.
    N, D = size(x)

    # latent variable z
    z ~ filldist(Normal(), D, N)

    # side note for the curious
    # we use the more concise filldist syntax partly for compactness, but also for compatibility with other AD
    # backends, see the [Turing Performance Tipps](https://turing.ml/dev/docs/using-turing/performancetips)
    # w = TV{2}(undef, D, D)
    # for d in 1:D
    #  w[d, :] ~ MvNormal(ones(D))
    # end

    # weights/loadings W
    w ~ filldist(Normal(), D, D)

    # mean offset
    m ~ MvNormal(ones(D))
    mu = (w * z .+ m)'
    for d in 1:D
        x[:, d] ~ MvNormal(mu[:, d], ones(N))
    end
end;


ppca = pPCA(expression_matrix)
chain_ppca = sample(ppca, NUTS(), 500);


# Extract parameter estimates for plotting - mean of posterior
w = reshape(mean(group(chain_ppca, :w))[:, 2], (n_genes, n_genes))
z = permutedims(reshape(mean(group(chain_ppca, :z))[:, 2], (n_genes, n_cells)))'
mu = mean(group(chain_ppca, :m))[:, 2]

X = w * z

df_rec = DataFrame(X', :auto)
df_rec[!, :cell] = 1:n_cells

@vlplot(
    :rect,
    x = "cell:o",
    color = :value,
    encoding = {
        y = {field = "variable", type = "nominal", sort = "-x", axis = {title = "gene"}}
    }
)(
    DataFrames.stack(df_rec, 1:n_genes)
)


let
    diff = X' - expression_matrix
    @assert mean(diff[:, 4]) < 0.5
    @assert mean(diff[:, 5]) < 0.5
    @assert mean(diff[:, 6]) < 0.5
end


df_pca = DataFrame(z', :auto)
rename!(df_pca, Symbol.(["z" * string(i) for i in collect(1:n_genes)]))
df_pca[!, :cell] = 1:n_cells

@vlplot(:rect, "cell:o", "variable:o", color = :value)(DataFrames.stack(df_pca, 1:n_genes))

df_pca[!, :type] = repeat([1, 2]; inner=n_cells ÷ 2)
@vlplot(:point, x = :z1, y = :z2, color = "type:n")(df_pca)


@model function pPCA_ARD(x, ::Type{TV}=Array{Float64}) where {TV}
    # Dimensionality of the problem.
    N, D = size(x)

    # latent variable z
    z ~ filldist(Normal(), D, N)

    # weights/loadings w with Automatic Relevance Determination part
    alpha ~ filldist(Gamma(1.0, 1.0), D)
    w ~ filldist(MvNormal(zeros(D), 1.0 ./ sqrt.(alpha)), D)

    mu = (w' * z)'

    tau ~ Gamma(1.0, 1.0)
    for d in 1:D
        x[:, d] ~ MvNormal(mu[:, d], 1.0 / sqrt(tau))
    end
end;


ppca_ARD = pPCA_ARD(expression_matrix)
chain_pccaARD = sample(ppca_ARD, NUTS(), 500)

StatsPlots.plot(group(chain_pccaARD, :alpha))


# Extract parameter estimates for plotting - mean of posterior
w = permutedims(reshape(mean(group(chain_pccaARD, :w))[:, 2], (n_genes, n_genes)))
z = permutedims(reshape(mean(group(chain_pccaARD, :z))[:, 2], (n_genes, n_cells)))'
α = mean(group(chain_pccaARD, :alpha))[:, 2]
α


alpha_indices = sortperm(α)[1:2]
X = w[alpha_indices, alpha_indices] * z[alpha_indices, :]

df_rec = DataFrame(X', :auto)
df_rec[!, :cell] = 1:n_cells
@vlplot(:rect, "cell:o", "variable:o", color = :value)(DataFrames.stack(df_rec, 1:2))

df_pre = DataFrame(z', :auto)
rename!(df_pre, Symbol.(["z" * string(i) for i in collect(1:n_genes)]))
df_pre[!, :cell] = 1:n_cells

@vlplot(:rect, "cell:o", "variable:o", color = :value)(DataFrames.stack(df_pre, 1:n_genes))

df_pre[!, :type] = repeat([1, 2]; inner=n_cells ÷ 2)
df_pre[!, :ard1] = df_pre[:, alpha_indices[1]]
df_pre[!, :ard2] = df_pre[:, alpha_indices[2]]
@vlplot(:point, x = :ard1, y = :ard2, color = "type:n")(df_pre)


# Example data set - generate synthetic gene expression data

# dataset available in RDatasets
data = dataset("datasets", "iris")
species = data[!, "Species"]

# we extract the four measured quantities
d = 4
dat = data[!, 1:d]
# and the number of measurements
n = size(dat)[1];


ppca = pPCA(dat)

# Here we use a different sampler, we don't always have to use NUTS:
# Hamiltonian Monte Carlo (HMC) sampler parameters
ϵ = 0.05
τ = 10
chain_ppca2 = sample(ppca, HMC(ϵ, τ), 1000)

# Extract parameter estimates for plotting - mean of posterior
w = permutedims(reshape(mean(group(chain_ppca2, :w))[:, 2], (d, d)))
z = permutedims(reshape(mean(group(chain_ppca2, :z))[:, 2], (d, n)))'
mu = mean(group(chain_ppca2, :m))[:, 2]

X = w * z
# X = w * z .+ mu

df_rec = DataFrame(X', :auto)
df_rec[!, :species] = species
@vlplot(:rect, "species:o", "variable:o", color = :value)(DataFrames.stack(df_rec, 1:d))

df_iris = DataFrame(z', :auto)
rename!(df_iris, Symbol.(["z" * string(i) for i in collect(1:d)]))
df_iris[!, :sample] = 1:n
df_iris[!, :species] = species

@vlplot(:point, x = :z1, y = :z2, color = "species:n")(df_iris)


## Introduce batch effect
batch = rand(Binomial(1, 0.5), 150)
effect = rand(Normal(2.4, 0.6), 150)
batch_dat = dat .+ batch .* effect

ppca_batch = pPCA(batch_dat)
chain_ppcaBatch = sample(ppca_batch, HMC(ϵ, τ), 1000)
describe(chain_ppcaBatch)[1]

z = permutedims(reshape(mean(group(chain_ppcaBatch, :z))[:, 2], (d, n)))'
df_pre = DataFrame(z', :auto)
rename!(df_pre, Symbol.(["z" * string(i) for i in collect(1:d)]))
df_pre[!, :sample] = 1:n
df_pre[!, :species] = species
df_pre[!, :batch] = batch

@vlplot(:point, x = :z1, y = :z2, color = "species:n", shape = :batch)(df_pre)


@model function pPCA_residual(x, batch, ::Type{TV}=Array{Float64}) where {TV}

    # Dimensionality of the problem.
    N, D = size(x)

    # latent variable z
    z ~ filldist(Normal(), D, N)

    # weights/loadings w
    w ~ filldist(Normal(), D, D)

    # covariate vector
    w_batch = TV{1}(undef, D)
    w_batch ~ MvNormal(ones(D))

    # mean offset
    m = TV{1}(undef, D)
    m ~ MvNormal(ones(D))
    mu = m .+ w * z + w_batch .* batch'

    for d in 1:D
        x[:, d] ~ MvNormal(mu'[:, d], ones(N))
    end
end;

ppca_residual = pPCA_residual(batch_dat, convert(Vector{Float64}, batch))
chain_ppcaResidual = sample(ppca_residual, HMC(ϵ, τ), 1000);


z = permutedims(reshape(mean(group(chain_ppcaResidual, :z))[:, 2], (d, n)))'
df_post = DataFrame(z', :auto)
rename!(df_post, Symbol.(["z" * string(i) for i in collect(1:d)]))
df_post[!, :sample] = 1:n
df_post[!, :species] = species
df_post[!, :batch] = batch

@vlplot(:point, x = :z1, y = :z2, color = "species:n", shape = :batch)(df_post)


## helper functions for Householder transform
function V_low_tri_plus_diag(Q::Int, V)
    for q in 1:Q
        V[:, q] = V[:, q] ./ sqrt(sum(V[:, q] .^ 2))
    end
    return (V)
end

function Householder(k::Int, V)
    v = V[:, k]
    sgn = sign(v[k])

    v[k] += sgn
    H = LinearAlgebra.I - (2.0 / dot(v, v) * (v * v'))
    H[k:end, k:end] = -1.0 * sgn .* H[k:end, k:end]

    return (H)
end

function H_prod_right(V)
    D, Q = size(V)

    H_prod = zeros(Real, D, D, Q + 1)
    H_prod[:, :, 1] = Diagonal(repeat([1.0], D))

    for q in 1:Q
        H_prod[:, :, q + 1] = Householder(Q - q + 1, V) * H_prod[:, :, q]
    end

    return (H_prod)
end

function orthogonal_matrix(D::Int64, Q::Int64, V)
    V = V_low_tri_plus_diag(Q, V)
    H_prod = H_prod_right(V)

    return (H_prod[:, 1:Q, Q + 1])
end

@model function pPCA_householder(x, K::Int, ::Type{T}=Float64) where {T}

    # Dimensionality of the problem.
    D, N = size(x)
    @assert K <= D

    # parameters
    sigma_noise ~ LogNormal(0.0, 0.5)
    v ~ filldist(Normal(0.0, 1.0), Int(D * K - K * (K - 1) / 2))
    sigma ~ Bijectors.ordered(MvLogNormal(MvNormal(ones(K))))

    v_mat = zeros(T, D, K)
    v_mat[tril!(trues(size(v_mat)))] .= v
    U = orthogonal_matrix(D, Q, v_mat)

    W = zeros(T, D, K)
    W += U * Diagonal(sigma)

    Kmat = zeros(T, D, D)
    Kmat += W * W'
    for d in 1:D
        Kmat[d, d] = Kmat[d, d] + sigma_noise^2 + 1e-12
    end
    L = LinearAlgebra.cholesky(Kmat).L

    for q in 1:Q
        r = sqrt.(sum(dot(v_mat[:, q], v_mat[:, q])))
        Turing.@addlogprob! (-log(r) * (D - q))
    end

    Turing.@addlogprob! -0.5 * sum(sigma .^ 2) + (D - Q - 1) * sum(log.(sigma))
    for qi in 1:Q
        for qj in (qi + 1):Q
            Turing.@addlogprob! log(sigma[Q - qi + 1]^2) - sigma[Q - qj + 1]^2
        end
    end
    Turing.@addlogprob! sum(log.(2.0 * sigma))

    L_full = zeros(T, D, D)
    L_full += L * transpose(L)
    # fix numerical instability (non-posdef matrix)
    for d in 1:D
        for k in (d + 1):D
            L_full[d, k] = L_full[k, d]
        end
    end

    return x ~ filldist(MvNormal(L_full), N)
end;

# Dimensionality of latent space
Random.seed!(1789);
Q = 2
n_samples = 700
ppca_householder = pPCA_householder(Matrix(dat)', Q)
chain_ppcaHouseholder = sample(ppca_householder, NUTS(), n_samples);

# Extract mean of v from chain
N, D = size(dat)
vv = mean(group(chain_ppcaHouseholder, :v))[:, 2]
v_mat = zeros(Real, D, Q)
v_mat[tril!(trues(size(v_mat)))] .= vv
sigma = mean(group(chain_ppcaHouseholder, :sigma))[:, 2]
U_n = orthogonal_matrix(D, Q, v_mat)
W_n = U_n * (LinearAlgebra.I(Q) .* sigma)

# Create array with projected values
z = W_n' * transpose(Matrix(dat))
df_post = DataFrame(convert(Array{Float64}, z)', :auto)
rename!(df_post, Symbol.(["z" * string(i) for i in collect(1:Q)]))
df_post[!, :sample] = 1:n
df_post[!, :species] = species

@vlplot(:point, x = :z1, y = :z2, color = "species:n")(df_post)


## Create data projections for each step of chain
vv = collect(get(chain_ppcaHouseholder, [:v]).v)
v_mat = zeros(Real, D, Q)
vv_mat = zeros(Float64, n_samples, D, Q)
for i in 1:n_samples
    index = BitArray(zeros(n_samples, D, Q))
    index[i, :, :] = tril!(trues(size(v_mat)))
    tmp = zeros(size(vv)[1])
    for j in 1:(size(vv)[1])
        tmp[j] = vv[j][i]
    end
    vv_mat[index] .= tmp
end

ss = collect(get(chain_ppcaHouseholder, [:sigma]).sigma)
sigma = zeros(Q, n_samples)
for d in 1:Q
    sigma[d, :] = Array(ss[d])
end

samples_raw = Array{Float64}(undef, Q, N, n_samples)
for i in 1:n_samples
    U_ni = orthogonal_matrix(D, Q, vv_mat[i, :, :])
    W_ni = U_ni * (LinearAlgebra.I(Q) .* sigma[:, i])
    z_n = W_ni' * transpose(Matrix(dat))
    samples_raw[:, :, i] = z_n
end

# initialize a 3D plot with 1 empty series
plt = plot(
    [100, 200, 300];
    xlim=(-4.00, 7.00),
    ylim=(-100.00, 0.00),
    group=["Setosa", "Versicolor", "Virginica"],
    markercolor=["red", "blue", "black"],
    title="Visualization",
    seriestype=:scatter,
)

anim = @animate for i in 1:n_samples
    scatter!(
        plt,
        samples_raw[1, 1:50, i],
        samples_raw[2, 1:50, i];
        color="red",
        seriesalpha=0.1,
        label="",
    )
    scatter!(
        plt,
        samples_raw[1, 51:100, i],
        samples_raw[2, 51:100, i];
        color="blue",
        seriesalpha=0.1,
        label="",
    )
    scatter!(
        plt,
        samples_raw[1, 101:150, i],
        samples_raw[2, 101:150, i];
        color="black",
        seriesalpha=0.1,
        label="",
    )
end
gif(anim, "anim_fps.gif"; fps=5)


let
    m1 = mean(samples_raw[1, 1:50, :])
    m2 = mean(samples_raw[1, 51:100, :])
    m3 = mean(samples_raw[1, 101:150, :])
    @assert m1 - m2 > 3
    @assert m1 - m3 > 3
    @assert m2 - m3 < 2
end


# kernel density estimate
using StatsPlots, KernelDensity
dens = kde((vec(samples_raw[1, :, :]), vec(samples_raw[2, :, :])))
StatsPlots.plot(dens)


if isdefined(Main, :TuringTutorials)
    Main.TuringTutorials.tutorial_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])
end

