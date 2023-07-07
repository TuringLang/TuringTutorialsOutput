

using Turing
using ReverseDiff
Turing.setadbackend(:reversediff)
using LinearAlgebra, FillArrays

# Packages for visualization
using DataFrames, StatsPlots, Measures

# Set a seed for reproducibility.
using Random
Random.seed!(1789);


n_genes = 9 # D
n_cells = 60 # N

# create a diagonal block like expression matrix, with some non-informative genes;
# not all features/genes are informative, some might just not differ very much between cells)
mat_exp = randn(n_genes, n_cells)
mat_exp[1:(n_genes ÷ 3), 1:(n_cells ÷ 2)] .+= 10
mat_exp[(2 * (n_genes ÷ 3) + 1):end, (n_cells ÷ 2 + 1):end] .+= 10


heatmap(
    mat_exp;
    c=:summer,
    colors=:value,
    xlabel="cell number",
    yflip=true,
    ylabel="gene feature",
    yticks=1:9,
    colorbar_title="expression",
)


@model function pPCA(X::AbstractMatrix{<:Real}, k::Int)
    # retrieve the dimension of input matrix X.
    N, D = size(X)

    # weights/loadings W
    W ~ filldist(Normal(), D, k)

    # latent variable z
    Z ~ filldist(Normal(), k, N)

    # mean offset
    μ ~ MvNormal(Eye(D))
    genes_mean = W * Z .+ reshape(μ, n_genes, 1)
    return X ~ arraydist([MvNormal(m, Eye(N)) for m in eachcol(genes_mean')])
end;


k = 2 # k is the dimension of the projected space, i.e. the number of principal components/axes of choice
ppca = pPCA(mat_exp', k) # instantiate the probabilistic model
chain_ppca = sample(ppca, NUTS(), 500);


size(chain_ppca) # (no. of iterations, no. of vars, no. of chains) = (500, 159, 1)


# Extract parameter estimates for predicting x - mean of posterior
W = reshape(mean(group(chain_ppca, :W))[:, 2], (n_genes, k))
Z = reshape(mean(group(chain_ppca, :Z))[:, 2], (k, n_cells))
μ = mean(group(chain_ppca, :μ))[:, 2]

mat_rec = W * Z .+ repeat(μ; inner=(1, n_cells))


heatmap(
    mat_rec;
    c=:summer,
    colors=:value,
    xlabel="cell number",
    yflip=true,
    ylabel="gene feature",
    yticks=1:9,
    colorbar_title="expression",
)


# let
#     diff_matrix = mat_exp .- mat_rec
#     @assert abs(mean(diff_matrix[:, 4])) <= 0.5 #0.327
#     @assert abs(mean(diff_matrix[:, 5])) <= 0.5 #0.390
#     @assert abs(mean(diff_matrix[:, 6])) <= 0.5 #0.326
# end


df_pca = DataFrame(Z', :auto)
rename!(df_pca, Symbol.(["z" * string(i) for i in collect(1:k)]))
df_pca[!, :type] = repeat([1, 2]; inner=n_cells ÷ 2)

scatter(df_pca[:, :z1], df_pca[:, :z2]; xlabel="z1", ylabel="z2", group=df_pca[:, :type])


@model function pPCA_ARD(X)
    # Dimensionality of the problem.
    N, D = size(X)

    # latent variable Z
    Z ~ filldist(Normal(), D, N)

    # weights/loadings w with Automatic Relevance Determination part
    α ~ filldist(Gamma(1.0, 1.0), D)
    W ~ filldist(MvNormal(zeros(D), 1.0 ./ sqrt.(α)), D)

    mu = (W' * Z)'

    tau ~ Gamma(1.0, 1.0)
    return X ~ arraydist([MvNormal(m, 1.0 / sqrt(tau)) for m in eachcol(mu)])
end;


ppca_ARD = pPCA_ARD(mat_exp') # instantiate the probabilistic model
chain_ppcaARD = sample(ppca_ARD, NUTS(), 500) # sampling
plot(group(chain_ppcaARD, :α); margin=6.0mm)


# Extract parameter mean estimates of the posterior
W = permutedims(reshape(mean(group(chain_ppcaARD, :W))[:, 2], (n_genes, n_genes)))
Z = permutedims(reshape(mean(group(chain_ppcaARD, :Z))[:, 2], (n_genes, n_cells)))'
α = mean(group(chain_ppcaARD, :α))[:, 2]
plot(α; label="α")


α_indices = sortperm(α)[1:2]
k = size(α_indices)[1]
X_rec = W[:, α_indices] * Z[α_indices, :]

df_rec = DataFrame(X_rec', :auto)
heatmap(
    X_rec;
    c=:summer,
    colors=:value,
    xlabel="cell number",
    yflip=true,
    ylabel="gene feature",
    yticks=1:9,
    colorbar_title="expression",
)


df_pro = DataFrame(Z[α_indices, :]', :auto)
rename!(df_pro, Symbol.(["z" * string(i) for i in collect(1:k)]))
df_pro[!, :type] = repeat([1, 2]; inner=n_cells ÷ 2)
scatter(
    df_pro[:, 1], df_pro[:, 2]; xlabel="z1", ylabel="z2", color=df_pro[:, "type"], label=""
)



