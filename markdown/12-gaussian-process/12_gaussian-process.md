---
redirect_from: "tutorials/12-gaussian-process/"
title: "Gaussian Process Latent Variable Model"
permalink: "/:collection/:name/"
---


# Gaussian Process Latent Variable Model

In a previous tutorial, we have discussed latent variable models, in particular probabilistic principal component analysis (pPCA).
Here, we show how we can extend the mapping provided by pPCA to non-linear mappings between input and output.
For more details about the Gaussian Process Latent Variable Model (GPLVM),
we refer the reader to the [original publication](https://jmlr.org/papers/v6/lawrence05a.html) and a [further extension](http://proceedings.mlr.press/v9/titsias10a/titsias10a.pdf).

In short, the GPVLM is a dimensionality reduction technique that allows us to embed a high-dimensional dataset in a lower-dimensional embedding.
Importantly, it provides the advantage that the linear mappings from the embedded space can be non-linearised through the use of Gaussian Processes.

Let's start by loading some dependencies.

```julia
using Turing
using AbstractGPs, Random

using LinearAlgebra
using VegaLite, DataFrames, StatsPlots, StatsBase
using RDatasets

Random.seed!(1789);
```




We demonstrate the GPLVM with a very small dataset: [Fisher's Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set).
This is mostly for reasons of run time, so the tutorial can be run quickly.
As you will see, one of the major drawbacks of using GPs is their speed,
although this is an active area of research.
We will briefly touch on some ways to speed things up at the end of this tutorial.
We transform the original data with non-linear operations in order to demonstrate the power of GPs to work on non-linear relationships, while keeping the problem reasonably small.

```julia
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
```




We will start out by demonstrating the basic similarity between pPCA (see the tutorial on this topic) and the GPLVM model.
Indeed, pPCA is basically equivalent to running the GPLVM model with an automatic relevance determination (ARD) linear kernel.

First, we re-introduce the pPCA model (see the tutorial on pPCA for details)

```julia
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
```




We define two different kernels, a simple linear kernel with an Automatic Relevance Determination transform and a
squared exponential kernel.

```julia
linear_kernel(α) = LinearKernel() ∘ ARDTransform(α)
sekernel(α, σ) = σ * SqExponentialKernel() ∘ ARDTransform(α);
```




And here is the GPLVM model.
We create separate models for the two types of kernel.

```julia
@model function GPLVM_linear(Y, K=4, ::Type{T}=Float64) where {T}

    # Dimensionality of the problem.
    N, D = size(Y)
    # K is the dimension of the latent space
    @assert K <= D
    noise = 1e-3

    # Priors
    α ~ MvLogNormal(MvNormal(zeros(K), I))
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
    α ~ MvLogNormal(MvNormal(zeros(K), I))
    σ ~ LogNormal(0.0, 1.0)
    Z ~ filldist(Normal(), K, N)
    mu ~ filldist(Normal(), N)

    kernel = sekernel(α, σ)

    gp = GP(mu, kernel)
    cv = cov(gp(ColVecs(Z), noise))
    return Y ~ filldist(MvNormal(mu, cv), D)
end;
```


```julia
# Standard GPs don't scale very well in n, so we use a small subsample for the purpose of this tutorial
n_data = 40
# number of features to use from dataset
n_features = 4
# latent dimension for GP case
ndim = 4;
```


```julia
ppca = pPCA(dat[1:n_data, 1:n_features])
chain_ppca = sample(ppca, NUTS(), 1000);
```


```julia
# we extract the posterior mean estimates of the parameters from the chain
w = reshape(mean(group(chain_ppca, :w))[:, 2], (n_features, n_features))
z = reshape(mean(group(chain_ppca, :z))[:, 2], (n_features, n_data))
X = w * z

df_pre = DataFrame(z', :auto)
rename!(df_pre, Symbol.(["z" * string(i) for i in collect(1:n_features)]))
df_pre[!, :type] = labels[1:n_data]
p_ppca = @vlplot(:point, x = :z1, y = :z2, color = "type:n")(df_pre)
```

![](figures/12_gaussian-process_8_1.png)



We can see that the pPCA fails to distinguish the groups.
In particular, the `setosa` species is not clearly separated from `versicolor` and `virginica`.
This is due to the non-linearities that we introduced, as without them the two groups can be clearly distinguished
using pPCA (see the pPCA tutorial).

Let's try the same with our linear kernel GPLVM model.

```julia
gplvm_linear = GPLVM_linear(dat[1:n_data, 1:n_features], ndim)

chain_linear = sample(gplvm_linear, NUTS(), 500)
# we extract the posterior mean estimates of the parameters from the chain
z_mean = reshape(mean(group(chain_linear, :Z))[:, 2], (n_features, n_data))
alpha_mean = mean(group(chain_linear, :α))[:, 2]
```

```
4-element Vector{Float64}:
 0.48246972574348507
 0.4547127054344807
 0.5679711122550067
 0.45665907429879266
```



```julia
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
```

```
[3, 1]
```


![](figures/12_gaussian-process_10_1.png)



We can see that similar to the pPCA case, the linear kernel GPLVM fails to distinguish between the two groups
(`setosa` on the one hand, and `virginica` and `verticolor` on the other).

Finally, we demonstrate that by changing the kernel to a non-linear function, we are able to separate the data again.

```julia
gplvm = GPLVM(dat[1:n_data, 1:n_features], ndim)

chain_gplvm = sample(gplvm, NUTS(), 500)
# we extract the posterior mean estimates of the parameters from the chain
z_mean = reshape(mean(group(chain_gplvm, :Z))[:, 2], (ndim, n_data))
alpha_mean = mean(group(chain_gplvm, :α))[:, 2]
```

```
4-element Vector{Float64}:
 0.17392648795292895
 0.8050969213863638
 0.17221569204259213
 0.15629298383156232
```



```julia
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
```

```
[2, 1]
```


![](figures/12_gaussian-process_12_1.png)




Now, the split between the two groups is visible again.

### Speeding up inference

Gaussian processes tend to be slow, as they naively require operations in the order of $O(n^3)$.
Here, we demonstrate a simple speedup using the Stheno library.
Speeding up Gaussian process inference is an active area of research.

```julia
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
    α ~ MvLogNormal(MvNormal(zeros(K), I))
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
```

```
GPLVM_sparse (generic function with 3 methods)
```



```julia
n_data = 50
gplvm_sparse = GPLVM_sparse(dat[1:n_data, :], ndim)

chain_gplvm_sparse = sample(gplvm_sparse, NUTS(), 500)
# we extract the posterior mean estimates of the parameters from the chain
z_mean = reshape(mean(group(chain_gplvm_sparse, :Z))[:, 2], (ndim, n_data))
alpha_mean = mean(group(chain_gplvm_sparse, :α))[:, 2]
```

```
4-element Vector{Float64}:
 0.8621189679377088
 0.1539115560431259
 0.13382160252466813
 0.48650121015252795
```



```julia
df_gplvm_sparse = DataFrame(z_mean', :auto)
rename!(df_gplvm_sparse, Symbol.(["z" * string(i) for i in collect(1:ndim)]))
df_gplvm_sparse[!, :sample] = 1:n_data
df_gplvm_sparse[!, :labels] = labels[1:n_data]
alpha_indices = sortperm(alpha_mean; rev=true)[1:2]
df_gplvm_sparse[!, :ard1] = z_mean[alpha_indices[1], :]
df_gplvm_sparse[!, :ard2] = z_mean[alpha_indices[2], :]
p_sparse = @vlplot(:point, x = :ard1, y = :ard2, color = "labels:n")(df_gplvm_sparse)
p_sparse
```

![](figures/12_gaussian-process_16_1.png)



Comparing the runtime, between the two versions, we can observe a clear speed-up with the sparse version.


## Appendix

These tutorials are a part of the TuringTutorials repository, found at: [https://github.com/TuringLang/TuringTutorials](https://github.com/TuringLang/TuringTutorials).

To locally run this tutorial, do the following commands:

```
using TuringTutorials
TuringTutorials.weave("12-gaussian-process", "12_gaussian-process.jmd")
```

Computer Information:

```
Julia Version 1.6.6
Commit b8708f954a (2022-03-28 07:17 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, znver2)
Environment:
  JULIA_CPU_THREADS = 16
  BUILDKITE_PLUGIN_JULIA_CACHE_DIR = /cache/julia-buildkite-plugin
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/7aa0085e-79a4-45f3-a5bd-9743c91cf3da

```

Package Information:

```
      Status `/cache/build/default-amdci4-6/julialang/turingtutorials/tutorials/12-gaussian-process/Project.toml`
  [99985d1d] AbstractGPs v0.5.12
  [a93c6f00] DataFrames v1.3.3
  [ce6b1742] RDatasets v0.7.7
  [2913bbd2] StatsBase v0.33.16
  [f3b207a7] StatsPlots v0.14.33
  [8188c328] Stheno v0.8.1
  [fce5fe82] Turing v0.21.1
  [112f6efa] VegaLite v2.6.0
  [37e2e46d] LinearAlgebra
  [9a3f8284] Random
```

And the full manifest:

```
      Status `/cache/build/default-amdci4-6/julialang/turingtutorials/tutorials/12-gaussian-process/Manifest.toml`
  [621f4979] AbstractFFTs v1.1.0
  [99985d1d] AbstractGPs v0.5.12
  [80f14c24] AbstractMCMC v4.0.0
  [7a57a42e] AbstractPPL v0.5.2
  [1520ce14] AbstractTrees v0.3.4
  [79e6a3ab] Adapt v3.3.3
  [0bf59076] AdvancedHMC v0.3.4
  [5b7e9947] AdvancedMH v0.6.7
  [576499cb] AdvancedPS v0.3.7
  [b5ca4192] AdvancedVI v0.1.4
  [dce04be8] ArgCheck v2.3.0
  [7d9fca2a] Arpack v0.5.3
  [4fba245c] ArrayInterface v5.0.7
  [4c555306] ArrayLayouts v0.8.6
  [13072b0f] AxisAlgorithms v1.0.1
  [39de3d68] AxisArrays v0.4.5
  [198e06fe] BangBang v0.3.36
  [9718e550] Baselet v0.1.1
  [76274a88] Bijectors v0.9.11
  [8e7c35d0] BlockArrays v0.16.16
  [336ed68f] CSV v0.10.4
  [324d7699] CategoricalArrays v0.10.5
  [082447d4] ChainRules v1.28.3
  [d360d2e6] ChainRulesCore v1.14.0
  [9e997f8a] ChangesOfVariables v0.1.2
  [aaaa29a8] Clustering v0.14.2
  [944b1d66] CodecZlib v0.7.0
  [35d6a980] ColorSchemes v3.17.1
  [3da002f7] ColorTypes v0.11.0
  [5ae59095] Colors v0.12.8
  [861a8166] Combinatorics v1.0.2
  [38540f10] CommonSolve v0.2.0
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v3.43.0
  [a33af91c] CompositionsBase v0.1.1
  [88cd18e8] ConsoleProgressMonitor v0.1.2
  [187b0558] ConstructionBase v1.3.0
  [d38c429a] Contour v0.5.7
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.10.0
  [a93c6f00] DataFrames v1.3.3
  [864edb3b] DataStructures v0.18.11
  [e2d170a0] DataValueInterfaces v1.0.0
  [e7dc6d0d] DataValues v0.4.13
  [244e2a9f] DefineSingletons v0.1.2
  [b429d917] DensityInterface v0.4.0
  [163ba53b] DiffResults v1.0.3
  [b552c78f] DiffRules v1.10.0
  [b4f34e82] Distances v0.10.7
  [31c24e10] Distributions v0.25.53
  [ced4e74d] DistributionsAD v0.6.38
  [ffbed154] DocStringExtensions v0.8.6
  [366bfd00] DynamicPPL v0.19.1
  [cad2338a] EllipticalSliceSampling v0.5.0
  [e2ba6199] ExprTools v0.1.8
  [c87230d0] FFMPEG v0.4.1
  [7a1cc6ca] FFTW v1.4.6
  [5789e2e9] FileIO v1.13.0
  [8fc22ac5] FilePaths v0.8.3
  [48062228] FilePathsBase v0.9.18
  [1a297f60] FillArrays v0.13.2
  [53c48c17] FixedPointNumbers v0.8.4
  [59287772] Formatting v0.4.2
  [f6369f11] ForwardDiff v0.10.26
  [d9f16b24] Functors v0.2.8
  [28b8d3ca] GR v0.64.2
  [5c1252a2] GeometryBasics v0.4.2
  [42e2da0e] Grisu v1.0.2
  [cd3eb016] HTTP v0.9.17
  [7869d1d1] IRTools v0.4.5
  [615f187c] IfElse v0.1.1
  [83e8ac13] IniFile v0.5.1
  [22cec73e] InitialValues v0.3.1
  [842dd82b] InlineStrings v1.1.2
  [505f98c9] InplaceOps v0.3.0
  [a98d9a8b] Interpolations v0.13.6
  [8197267c] IntervalSets v0.6.1
  [3587e190] InverseFunctions v0.1.3
  [41ab1584] InvertedIndices v1.1.0
  [92d709cd] IrrationalConstants v0.1.1
  [c8e1da08] IterTools v1.4.0
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.4.1
  [682c06a0] JSON v0.21.3
  [7d188eb4] JSONSchema v0.3.4
  [5ab0869b] KernelDensity v0.6.3
  [ec8451be] KernelFunctions v0.10.38
  [8ac3fa9e] LRUCache v1.3.0
  [b964fa9f] LaTeXStrings v1.3.0
  [23fbe1c1] Latexify v0.15.14
  [1d6d02ad] LeftChildRightSiblingTrees v0.1.3
  [6f1fad26] Libtask v0.7.0
  [2ab3a3ac] LogExpFunctions v0.3.12
  [e6f89c97] LoggingExtras v0.4.7
  [c7f686f2] MCMCChains v5.1.1
  [be115224] MCMCDiagnosticTools v0.1.3
  [e80e1ace] MLJModelInterface v1.4.2
  [1914dd2f] MacroTools v0.5.9
  [dbb5928d] MappedArrays v0.4.1
  [739be429] MbedTLS v1.0.3
  [442fdcdd] Measures v0.3.1
  [128add7d] MicroCollections v0.1.2
  [e1d29d7a] Missings v1.0.2
  [78c3b35d] Mocking v0.7.3
  [6f286f6a] MultivariateStats v0.9.1
  [872c559c] NNlib v0.8.4
  [77ba4419] NaNMath v1.0.0
  [86f7a689] NamedArrays v0.9.6
  [c020b1a1] NaturalSort v1.0.0
  [b8a86587] NearestNeighbors v0.4.10
  [2bd173c7] NodeJS v1.3.0
  [510215fc] Observables v0.4.0
  [6fe1bfb0] OffsetArrays v1.10.8
  [bac558e1] OrderedCollections v1.4.1
  [90014a1f] PDMats v0.11.7
  [69de0a69] Parsers v2.3.0
  [ccf2f8ad] PlotThemes v3.0.0
  [995b91a9] PlotUtils v1.2.0
  [91a5bcdd] Plots v1.27.6
  [2dfb63ee] PooledArrays v1.4.1
  [21216c6a] Preferences v1.3.0
  [08abe8d2] PrettyTables v1.3.1
  [33c8b6b6] ProgressLogging v0.1.4
  [92933f4c] ProgressMeter v1.7.2
  [1fd47b50] QuadGK v2.4.2
  [df47a6cb] RData v0.8.3
  [ce6b1742] RDatasets v0.7.7
  [b3c3ace0] RangeArrays v0.3.2
  [c84ed2f1] Ratios v0.4.3
  [c1ae055f] RealDot v0.1.0
  [3cdcf5f2] RecipesBase v1.2.1
  [01d81517] RecipesPipeline v0.5.2
  [731186ca] RecursiveArrayTools v2.26.3
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v0.1.3
  [ae029012] Requires v1.3.0
  [79098fc4] Rmath v0.7.0
  [f2b01f46] Roots v1.4.1
  [0bca4576] SciMLBase v1.31.0
  [30f210dd] ScientificTypesBase v3.0.0
  [6c6a2e73] Scratch v1.1.0
  [91c51154] SentinelArrays v1.3.12
  [efcf1570] Setfield v0.7.1
  [992d4aef] Showoff v1.0.3
  [a2af1166] SortingAlgorithms v1.0.1
  [276daf66] SpecialFunctions v2.1.4
  [171d559e] SplittablesBase v0.1.14
  [aedffcd0] Static v0.6.1
  [90137ffa] StaticArrays v1.4.4
  [64bff920] StatisticalTraits v3.0.0
  [82ae8749] StatsAPI v1.2.2
  [2913bbd2] StatsBase v0.33.16
  [4c63d2b9] StatsFuns v0.9.18
  [f3b207a7] StatsPlots v0.14.33
  [8188c328] Stheno v0.8.1
  [09ab397b] StructArrays v0.6.5
  [ab02a1b2] TableOperations v1.2.0
  [3783bdb8] TableTraits v1.0.1
  [382cd787] TableTraitsUtils v1.0.2
  [bd369af6] Tables v1.7.0
  [62fd8b95] TensorCore v0.1.1
  [5d786b92] TerminalLoggers v0.1.5
  [f269a46b] TimeZones v1.7.3
  [9f7883ad] Tracker v0.2.20
  [3bb67fe8] TranscodingStreams v0.9.6
  [28d57a85] Transducers v0.4.73
  [a2a6695c] TreeViews v0.3.0
  [fce5fe82] Turing v0.21.1
  [30578b45] URIParser v0.4.1
  [5c2747f8] URIs v1.3.0
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
  [41fe7b60] Unzip v0.1.2
  [239c3e63] Vega v2.3.0
  [112f6efa] VegaLite v2.6.0
  [ea10d353] WeakRefStrings v1.4.2
  [cc8bc4a8] Widgets v0.6.5
  [efce3f68] WoodburyMatrices v0.5.5
  [700de1a5] ZygoteRules v0.2.2
  [68821587] Arpack_jll v3.5.0+3
  [6e34b625] Bzip2_jll v1.0.8+0
  [83423d85] Cairo_jll v1.16.1+1
  [5ae413db] EarCut_jll v2.2.3+0
  [2e619515] Expat_jll v2.4.8+0
  [b22a6f82] FFMPEG_jll v4.4.0+0
  [f5851436] FFTW_jll v3.3.10+0
  [a3f928ae] Fontconfig_jll v2.13.93+0
  [d7e528f0] FreeType2_jll v2.10.4+0
  [559328eb] FriBidi_jll v1.0.10+0
  [0656b61e] GLFW_jll v3.3.6+0
  [d2c73de3] GR_jll v0.64.2+0
  [78b55507] Gettext_jll v0.21.0+0
  [7746bdde] Glib_jll v2.68.3+2
  [3b182d85] Graphite2_jll v1.3.14+0
  [2e76f6c2] HarfBuzz_jll v2.8.1+1
  [1d5cc7b8] IntelOpenMP_jll v2018.0.3+2
  [aacddb02] JpegTurbo_jll v2.1.2+0
  [c1c5ebd0] LAME_jll v3.100.1+0
  [88015f11] LERC_jll v3.0.0+1
  [dd4b983a] LZO_jll v2.10.1+0
  [e9f186c6] Libffi_jll v3.2.2+1
  [d4300ac3] Libgcrypt_jll v1.8.7+0
  [7e76a0d4] Libglvnd_jll v1.3.0+3
  [7add5ba3] Libgpg_error_jll v1.42.0+0
  [94ce4f54] Libiconv_jll v1.16.1+1
  [4b2f31a3] Libmount_jll v2.35.0+0
  [89763e89] Libtiff_jll v4.3.0+1
  [38a345b3] Libuuid_jll v2.36.0+0
  [856f044c] MKL_jll v2022.0.0+0
  [e7412a2a] Ogg_jll v1.3.5+1
  [458c3c95] OpenSSL_jll v1.1.14+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.2+0
  [2f80f16e] PCRE_jll v8.44.0+0
  [30392449] Pixman_jll v0.40.1+0
  [ea2cea3b] Qt5Base_jll v5.15.3+1
  [f50d1b31] Rmath_jll v0.3.0+0
  [a2964d1f] Wayland_jll v1.19.0+0
  [2381bf8a] Wayland_protocols_jll v1.25.0+0
  [02c8fc9c] XML2_jll v2.9.12+0
  [aed1982a] XSLT_jll v1.1.34+0
  [4f6342f7] Xorg_libX11_jll v1.6.9+4
  [0c0b7dd1] Xorg_libXau_jll v1.0.9+4
  [935fb764] Xorg_libXcursor_jll v1.2.0+4
  [a3789734] Xorg_libXdmcp_jll v1.1.3+4
  [1082639a] Xorg_libXext_jll v1.3.4+4
  [d091e8ba] Xorg_libXfixes_jll v5.0.3+4
  [a51aa0fd] Xorg_libXi_jll v1.7.10+4
  [d1454406] Xorg_libXinerama_jll v1.1.4+4
  [ec84b674] Xorg_libXrandr_jll v1.5.2+4
  [ea2f1a96] Xorg_libXrender_jll v0.9.10+4
  [14d82f49] Xorg_libpthread_stubs_jll v0.1.0+3
  [c7cfdc94] Xorg_libxcb_jll v1.13.0+3
  [cc61e674] Xorg_libxkbfile_jll v1.1.0+4
  [12413925] Xorg_xcb_util_image_jll v0.4.0+1
  [2def613f] Xorg_xcb_util_jll v0.4.0+1
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.0+1
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.9+1
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.1+1
  [35661453] Xorg_xkbcomp_jll v1.4.2+4
  [33bec58e] Xorg_xkeyboard_config_jll v2.27.0+4
  [c5fb5394] Xorg_xtrans_jll v1.4.0+3
  [3161d3a3] Zstd_jll v1.5.2+0
  [0ac62f75] libass_jll v0.15.1+0
  [f638f0a6] libfdk_aac_jll v2.0.2+0
  [b53b4c65] libpng_jll v1.6.38+0
  [f27f6e37] libvorbis_jll v1.3.7+1
  [1270edf5] x264_jll v2021.5.5+0
  [dfaa095f] x265_jll v3.5.0+0
  [d8fb68d0] xkbcommon_jll v0.9.1+5
  [0dad84c5] ArgTools
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [ade2ca70] Dates
  [8bb1440f] DelimitedFiles
  [8ba89e20] Distributed
  [f43a241f] Downloads
  [9fa8497b] Future
  [b77e0a4c] InteractiveUtils
  [4af54fe1] LazyArtifacts
  [b27032c2] LibCURL
  [76f85450] LibGit2
  [8f399da3] Libdl
  [37e2e46d] LinearAlgebra
  [56ddb016] Logging
  [d6f4376e] Markdown
  [a63ad114] Mmap
  [ca575930] NetworkOptions
  [44cfe95a] Pkg
  [de0858da] Printf
  [3fa0cd96] REPL
  [9a3f8284] Random
  [ea8e919c] SHA
  [9e88b42a] Serialization
  [1a1011a3] SharedArrays
  [6462fe0b] Sockets
  [2f01184e] SparseArrays
  [10745b16] Statistics
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML
  [a4e569a6] Tar
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll
  [deac9b47] LibCURL_jll
  [29816b5a] LibSSH2_jll
  [c8ffd9c3] MbedTLS_jll
  [14a3606d] MozillaCACerts_jll
  [4536629a] OpenBLAS_jll
  [05823500] OpenLibm_jll
  [83775a58] Zlib_jll
  [8e850ede] nghttp2_jll
  [3f19e933] p7zip_jll
```

