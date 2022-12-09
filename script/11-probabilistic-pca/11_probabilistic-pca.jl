
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


@model function pPCA(x, k)
    # function pPCA(k::Int64, x::Type{TV}=Array{Float64}) where {TV}
    # retrieve the dimension of input matrix x.
    N, D = size(x)

    # latent variable z
    z ~ filldist(Normal(), k, N)

    # side note for the curious
    # we use the more concise filldist syntax partly for compactness, but also for compatibility with other AD
    # backends, see the [Turing Performance Tipps](https://turing.ml/dev/docs/using-turing/performancetips)
    # w = TV{2}(undef, D, D)
    # for d in 1:D
    #  w[d, :] ~ MvNormal(ones(D))
    # end

    # weights/loadings W
    w ~ filldist(Normal(), D, k)

    # mean offset
    m ~ MvNormal(ones(D))
    for d in 1:D
        mu_d = ((w * z)[d, :] .+ m[d]) # mu_d is an N x 1 vector representing the d-th feature values for all instances.
        x[:, d] ~ MvNormal(mu_d, ones(N))
    end
end;


k = 2 # k is the dimension of the projected space, i.e. the number of principal components/axes of choice
ppca = pPCA(expression_matrix, k) # instantiate the probabilistic model
chain_ppca = sample(ppca, NUTS(), 500);


size(chain_ppca)


plot(chain_ppca[:, 1, :]; xlabel="iteration_no", ylabel="sample_value")


# Extract parameter estimates for predicting x - mean of posterior
w = reshape(mean(group(chain_ppca, :w))[:, 2], (n_genes, k))
z = permutedims(reshape(mean(group(chain_ppca, :z))[:, 2], (k, n_cells)))'
mu = mean(group(chain_ppca, :m))[:, 2]

X = w * z .+ mu

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
    diff_matrix = X' - expression_matrix
    @assert abs(mean(diff_matrix[:, 4])) < 0.5
    @assert abs(mean(diff_matrix[:, 5])) < 0.5
    @assert abs(mean(diff_matrix[:, 6])) < 0.5
end


df_pca = DataFrame(z', :auto)
rename!(df_pca, Symbol.(["z" * string(i) for i in collect(1:k)]))
df_pca[!, :cell] = 1:n_cells

@vlplot(:rect, "cell:o", "variable:o", color = :value)(DataFrames.stack(df_pca, 1:k))
scatter(df_pca[:, :z1], df_pca[:, :z2]; xlabel="z1", ylabel="z2")

df_pca[!, :type] = repeat([1, 2]; inner=n_cells ÷ 2)
@vlplot(:point, x = :z1, y = :z2, color = "type:n")(df_pca)


@model function pPCA_ARD(x, ::Type{TV}=Array{Float64}) where {TV}
    # retrieve the dimension of input matrix x.
    N, D = size(x)

    # latent variable z
    z ~ filldist(Normal(), D, N)

    # Determine the number of loadings, i.e. number of columns of w, with Automatic Relevance Determination
    alpha ~ filldist(Gamma(1.0, 1.0), D)
    w ~ filldist(MvNormal(zeros(D), 1.0 ./ sqrt.(alpha)), D)

    mu = (w' * z)'

    tau ~ Gamma(1.0, 1.0)
    for d in 1:D
        x[:, d] ~ MvNormal(mu[:, d], 1.0 / sqrt(tau))
    end
end;


ppca_ARD = pPCA_ARD(expression_matrix) # instantiate the probabilistic model
chain_ppcaARD = sample(ppca_ARD, NUTS(), 500)

StatsPlots.plot(group(chain_ppcaARD, :alpha))


# Extract parameter mean estimates of the posterior
w = permutedims(reshape(mean(group(chain_ppcaARD, :w))[:, 2], (n_genes, n_genes)))
z = permutedims(reshape(mean(group(chain_ppcaARD, :z))[:, 2], (n_genes, n_cells)))'
α = mean(group(chain_ppcaARD, :alpha))[:, 2]
plot(α; label="alpha")


alpha_indices = sortperm(α)[1:2]
k = size(alpha_indices)[1]
X_rec = w[:, alpha_indices] * z[alpha_indices, :] # extract the loading vectors in w and the corresponding 

df_rec = DataFrame(X_rec', :auto)
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


df_pro = DataFrame(z[alpha_indices, :]', :auto)
rename!(df_pro, Symbol.(["z" * string(i) for i in collect(1:k)]))
df_pro[!, :cell] = 1:n_cells

df_pro[!, :type] = repeat([1, 2]; inner=n_cells ÷ 2)
scatter(
    df_pro[:, 1], df_pro[:, 2]; xlabel="z1", ylabel="z2", color=df_pro[:, "type"], label=""
)

