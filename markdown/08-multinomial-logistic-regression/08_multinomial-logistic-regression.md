---
redirect_from: "tutorials/8-multinomiallogisticregression/"
title: "Bayesian Multinomial Logistic Regression"
permalink: "/tutorials/:name/"
---


[Multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression) is an extension of logistic regression. Logistic regression is used to model problems in which there are exactly two possible discrete outcomes. Multinomial logistic regression is used to model problems in which there are two or more possible discrete outcomes.

In our example, we'll be using the iris dataset. The iris multiclass problem aims to predict the species of a flower given measurements (in centimeters) of sepal length and width and petal length and width. There are three possible species: Iris setosa, Iris versicolor, and Iris virginica.

To start, let's import all the libraries we'll need.

```julia
# Load Turing.
using Turing

# Load RDatasets.
using RDatasets

# Load StatsPlots for visualizations and diagnostics.
using StatsPlots

# Functionality for splitting and normalizing the data.
using MLDataUtils: shuffleobs, splitobs, rescale!

# We need a softmax function which is provided by NNlib.
using NNlib: softmax

# Functionality for constructing arrays with identical elements efficiently.
using FillArrays

# Functionality for working with scaled identity matrices.
using LinearAlgebra

# Set a seed for reproducibility.
using Random
Random.seed!(0);
```




## Data Cleaning & Set Up

Now we're going to import our dataset. Twenty rows of the dataset are shown below so you can get a good feel for what kind of data we have.

```julia
# Import the "iris" dataset.
data = RDatasets.dataset("datasets", "iris");

# Show twenty random rows.
data[rand(1:size(data, 1), 20), :]
```

```
20×5 DataFrame
 Row │ SepalLength  SepalWidth  PetalLength  PetalWidth  Species
     │ Float64      Float64     Float64      Float64     Cat…
─────┼──────────────────────────────────────────────────────────────
   1 │         5.7         2.8          4.1         1.3  versicolor
   2 │         5.1         3.5          1.4         0.3  setosa
   3 │         5.4         3.9          1.3         0.4  setosa
   4 │         7.6         3.0          6.6         2.1  virginica
   5 │         5.0         3.5          1.6         0.6  setosa
   6 │         5.0         3.6          1.4         0.2  setosa
   7 │         5.5         2.4          3.8         1.1  versicolor
   8 │         6.1         2.6          5.6         1.4  virginica
  ⋮  │      ⋮           ⋮            ⋮           ⋮           ⋮
  14 │         5.1         3.3          1.7         0.5  setosa
  15 │         6.7         3.1          4.7         1.5  versicolor
  16 │         6.2         2.2          4.5         1.5  versicolor
  17 │         4.7         3.2          1.3         0.2  setosa
  18 │         6.4         2.9          4.3         1.3  versicolor
  19 │         5.7         3.0          4.2         1.2  versicolor
  20 │         4.3         3.0          1.1         0.1  setosa
                                                      5 rows omitted
```





In this data set, the outcome `Species` is currently coded as a string. We convert it to a numerical value by using indices `1`, `2`, and `3` to indicate species `setosa`, `versicolor`, and `virginica`, respectively.

```julia
# Recode the `Species` column.
species = ["setosa", "versicolor", "virginica"]
data[!, :Species_index] = indexin(data[!, :Species], species)

# Show twenty random rows of the new species columns
data[rand(1:size(data, 1), 20), [:Species, :Species_index]]
```

```
20×2 DataFrame
 Row │ Species     Species_index
     │ Cat…        Union…
─────┼───────────────────────────
   1 │ virginica   3
   2 │ versicolor  2
   3 │ virginica   3
   4 │ setosa      1
   5 │ setosa      1
   6 │ versicolor  2
   7 │ setosa      1
   8 │ setosa      1
  ⋮  │     ⋮             ⋮
  14 │ virginica   3
  15 │ virginica   3
  16 │ versicolor  2
  17 │ setosa      1
  18 │ versicolor  2
  19 │ setosa      1
  20 │ setosa      1
                   5 rows omitted
```





After we've done that tidying, it's time to split our dataset into training and testing sets, and separate the features and target from the data. Additionally, we must rescale our feature variables so that they are centered around zero by subtracting each column by the mean and dividing it by the standard deviation. Without this step, Turing's sampler will have a hard time finding a place to start searching for parameter estimates.

```julia
# Split our dataset 50%/50% into training/test sets.
trainset, testset = splitobs(shuffleobs(data), 0.5)

# Define features and target.
features = [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]
target = :Species_index

# Turing requires data in matrix and vector form.
train_features = Matrix(trainset[!, features])
test_features = Matrix(testset[!, features])
train_target = trainset[!, target]
test_target = testset[!, target]

# Standardize the features.
μ, σ = rescale!(train_features; obsdim=1)
rescale!(test_features, μ, σ; obsdim=1);
```




## Model Declaration

Finally, we can define our model `logistic_regression`. It is a function that takes three arguments where

  - `x` is our set of independent variables;
  - `y` is the element we want to predict;
  - `σ` is the standard deviation we want to assume for our priors.

We select the `setosa` species as the baseline class (the choice does not matter). Then we create the intercepts and vectors of coefficients for the other classes against that baseline. More concretely, we create scalar intercepts `intercept_versicolor` and `intersept_virginica` and coefficient vectors `coefficients_versicolor` and `coefficients_virginica` with four coefficients each for the features `SepalLength`, `SepalWidth`, `PetalLength` and `PetalWidth`. We assume a normal distribution with mean zero and standard deviation `σ` as prior for each scalar parameter. We want to find the posterior distribution of these, in total ten, parameters to be able to predict the species for any given set of features.

```julia
# Bayesian multinomial logistic regression
@model function logistic_regression(x, y, σ)
    n = size(x, 1)
    length(y) == n ||
        throw(DimensionMismatch("number of observations in `x` and `y` is not equal"))

    # Priors of intercepts and coefficients.
    intercept_versicolor ~ Normal(0, σ)
    intercept_virginica ~ Normal(0, σ)
    coefficients_versicolor ~ MvNormal(Zeros(4), σ^2 * I)
    coefficients_virginica ~ MvNormal(Zeros(4), σ^2 * I)

    # Compute the likelihood of the observations.
    values_versicolor = intercept_versicolor .+ x * coefficients_versicolor
    values_virginica = intercept_virginica .+ x * coefficients_virginica
    for i in 1:n
        # the 0 corresponds to the base category `setosa`
        v = softmax([0, values_versicolor[i], values_virginica[i]])
        y[i] ~ Categorical(v)
    end
end;
```




## Sampling

Now we can run our sampler. This time we'll use [`NUTS`](https://turinglang.org/stable/docs/library/#Turing.Inference.NUTS) to sample from our posterior.

```julia
m = logistic_regression(train_features, train_target, 1)
chain = sample(m, NUTS(), MCMCThreads(), 1_500, 3)
```

```
Chains MCMC chain (1500×22×3 Array{Float64, 3}):

Iterations        = 751:1:2250
Number of chains  = 3
Samples per chain = 1500
Wall duration     = 10.03 seconds
Compute duration  = 9.59 seconds
parameters        = intercept_versicolor, intercept_virginica, coefficients
_versicolor[1], coefficients_versicolor[2], coefficients_versicolor[3], coe
fficients_versicolor[4], coefficients_virginica[1], coefficients_virginica[
2], coefficients_virginica[3], coefficients_virginica[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, 
tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
                  parameters      mean       std      mcse    ess_bulk    e
ss_ ⋯
                      Symbol   Float64   Float64   Float64     Float64     
Flo ⋯

        intercept_versicolor    1.0477    0.4932    0.0074   4514.4152   33
51. ⋯
         intercept_virginica   -0.5690    0.6661    0.0108   3775.0247   28
32. ⋯
  coefficients_versicolor[1]    1.3133    0.6225    0.0099   4013.1478   33
04. ⋯
  coefficients_versicolor[2]   -1.2898    0.4954    0.0075   4420.5076   34
14. ⋯
  coefficients_versicolor[3]    0.9844    0.7445    0.0110   4565.8354   32
07. ⋯
  coefficients_versicolor[4]    0.2024    0.7071    0.0111   4055.9663   32
80. ⋯
   coefficients_virginica[1]    0.7299    0.6524    0.0100   4255.7240   35
61. ⋯
   coefficients_virginica[2]   -0.8287    0.6181    0.0091   4614.0632   32
11. ⋯
   coefficients_virginica[3]    2.3222    0.8301    0.0114   5344.1340   31
07. ⋯
   coefficients_virginica[4]    2.7702    0.7677    0.0113   4682.8793   34
92. ⋯
                                                               3 columns om
itted

Quantiles
                  parameters      2.5%     25.0%     50.0%     75.0%     97
.5% ⋯
                      Symbol   Float64   Float64   Float64   Float64   Floa
t64 ⋯

        intercept_versicolor    0.1163    0.7173    1.0222    1.3795    2.0
550 ⋯
         intercept_virginica   -1.8845   -1.0128   -0.5703   -0.1242    0.7
471 ⋯
  coefficients_versicolor[1]    0.0794    0.9045    1.3010    1.7287    2.5
682 ⋯
  coefficients_versicolor[2]   -2.3140   -1.6143   -1.2700   -0.9443   -0.3
886 ⋯
  coefficients_versicolor[3]   -0.4464    0.4730    0.9898    1.4901    2.4
603 ⋯
  coefficients_versicolor[4]   -1.1916   -0.2672    0.2075    0.6672    1.6
070 ⋯
   coefficients_virginica[1]   -0.5460    0.2818    0.7215    1.1701    1.9
876 ⋯
   coefficients_virginica[2]   -2.0143   -1.2505   -0.8245   -0.4056    0.3
818 ⋯
   coefficients_virginica[3]    0.7478    1.7608    2.3190    2.8594    3.9
546 ⋯
   coefficients_virginica[4]    1.2624    2.2629    2.7537    3.2938    4.3
052 ⋯
```





Since we ran multiple chains, we may as well do a spot check to make sure each chain converges around similar points.

```julia
plot(chain)
```

![](figures/08_multinomial-logistic-regression_7_1.png)



Looks good!

We can also use the `corner` function from MCMCChains to show the distributions of the various parameters of our multinomial logistic regression. The corner function requires MCMCChains and StatsPlots.

```julia
corner(
    chain,
    MCMCChains.namesingroup(chain, :coefficients_versicolor);
    label=[string(i) for i in 1:4],
)
```

![](figures/08_multinomial-logistic-regression_8_1.png)

```julia
corner(
    chain,
    MCMCChains.namesingroup(chain, :coefficients_virginica);
    label=[string(i) for i in 1:4],
)
```

![](figures/08_multinomial-logistic-regression_9_1.png)



Fortunately the corner plots appear to demonstrate unimodal distributions for each of our parameters, so it should be straightforward to take the means of each parameter's sampled values to estimate our model to make predictions.

## Making Predictions

How do we test how well the model actually predicts which of the three classes an iris flower belongs to? We need to build a `prediction` function that takes the test dataset and runs it through the average parameter calculated during sampling.

The `prediction` function below takes a `Matrix` and a `Chains` object. It computes the mean of the sampled parameters and calculates the species with the highest probability for each observation. Note that we do not have to evaluate the `softmax` function since it does not affect the order of its inputs.

```julia
function prediction(x::Matrix, chain)
    # Pull the means from each parameter's sampled values in the chain.
    intercept_versicolor = mean(chain, :intercept_versicolor)
    intercept_virginica = mean(chain, :intercept_virginica)
    coefficients_versicolor = [
        mean(chain, k) for k in MCMCChains.namesingroup(chain, :coefficients_versicolor)
    ]
    coefficients_virginica = [
        mean(chain, k) for k in MCMCChains.namesingroup(chain, :coefficients_virginica)
    ]

    # Compute the index of the species with the highest probability for each observation.
    values_versicolor = intercept_versicolor .+ x * coefficients_versicolor
    values_virginica = intercept_virginica .+ x * coefficients_virginica
    species_indices = [
        argmax((0, x, y)) for (x, y) in zip(values_versicolor, values_virginica)
    ]

    return species_indices
end;
```




Let's see how we did! We run the test matrix through the prediction function, and compute the accuracy for our prediction.

```julia
# Make the predictions.
predictions = prediction(test_features, chain)

# Calculate accuracy for our test set.
mean(predictions .== testset[!, :Species_index])
```

```
0.92
```





Perhaps more important is to see the accuracy per class.

```julia
for s in 1:3
    rows = testset[!, :Species_index] .== s
    println("Number of `", species[s], "`: ", count(rows))
    println(
        "Percentage of `",
        species[s],
        "` predicted correctly: ",
        mean(predictions[rows] .== testset[rows, :Species_index]),
    )
end
```

```
Number of `setosa`: 26
Percentage of `setosa` predicted correctly: 1.0
Number of `versicolor`: 25
Percentage of `versicolor` predicted correctly: 0.8
Number of `virginica`: 24
Percentage of `virginica` predicted correctly: 0.9583333333333334
```





This tutorial has demonstrated how to use Turing to perform Bayesian multinomial logistic regression.


## Appendix

These tutorials are a part of the TuringTutorials repository, found at: [https://github.com/TuringLang/TuringTutorials](https://github.com/TuringLang/TuringTutorials).

To locally run this tutorial, do the following commands:

```
using TuringTutorials
TuringTutorials.weave("08-multinomial-logistic-regression", "08_multinomial-logistic-regression.jmd")
```

Computer Information:

```
Julia Version 1.9.2
Commit e4ee485e909 (2023-07-05 09:39 UTC)
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver2)
  Threads: 1 on 16 virtual cores
Environment:
  JULIA_CPU_THREADS = 16
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/7aa0085e-79a4-45f3-a5bd-9743c91cf3da
  JULIA_IMAGE_THREADS = 1

```

Package Information:

```
Status `/cache/build/default-amdci4-3/julialang/turingtutorials/tutorials/08-multinomial-logistic-regression/Project.toml`
⌅ [1a297f60] FillArrays v1.0.0
  [cc2ba9b6] MLDataUtils v0.5.4
⌃ [872c559c] NNlib v0.9.1
  [ce6b1742] RDatasets v0.7.7
⌃ [f3b207a7] StatsPlots v0.15.5
⌃ [fce5fe82] Turing v0.26.2
  [37e2e46d] LinearAlgebra
  [9a3f8284] Random
Info Packages marked with ⌃ and ⌅ have new versions available, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated`
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

And the full manifest:

```
Status `/cache/build/default-amdci4-3/julialang/turingtutorials/tutorials/08-multinomial-logistic-regression/Manifest.toml`
  [47edcb42] ADTypes v0.1.5
  [621f4979] AbstractFFTs v1.4.0
  [80f14c24] AbstractMCMC v4.4.2
⌅ [7a57a42e] AbstractPPL v0.5.4
  [1520ce14] AbstractTrees v0.4.4
  [79e6a3ab] Adapt v3.6.2
  [0bf59076] AdvancedHMC v0.4.6
⌃ [5b7e9947] AdvancedMH v0.7.4
  [576499cb] AdvancedPS v0.4.3
  [b5ca4192] AdvancedVI v0.2.4
  [dce04be8] ArgCheck v2.3.0
  [7d9fca2a] Arpack v0.5.4
  [4fba245c] ArrayInterface v7.4.11
  [a9b6321e] Atomix v0.1.0
  [13072b0f] AxisAlgorithms v1.0.1
  [39de3d68] AxisArrays v0.4.6
  [198e06fe] BangBang v0.3.39
  [9718e550] Baselet v0.1.1
⌅ [76274a88] Bijectors v0.12.8
  [d1d4a3ce] BitFlags v0.1.7
  [fa961155] CEnum v0.4.2
  [336ed68f] CSV v0.10.11
  [49dc2e85] Calculus v0.5.1
  [324d7699] CategoricalArrays v0.10.8
⌃ [082447d4] ChainRules v1.52.0
  [d360d2e6] ChainRulesCore v1.16.0
  [9e997f8a] ChangesOfVariables v0.1.8
  [aaaa29a8] Clustering v0.15.3
⌃ [944b1d66] CodecZlib v0.7.1
  [35d6a980] ColorSchemes v3.21.0
  [3da002f7] ColorTypes v0.11.4
  [c3611d14] ColorVectorSpace v0.9.10
  [5ae59095] Colors v0.12.10
  [861a8166] Combinatorics v1.0.2
  [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v4.7.0
  [a33af91c] CompositionsBase v0.1.2
⌃ [f0e56b4a] ConcurrentUtilities v2.2.0
  [88cd18e8] ConsoleProgressMonitor v0.1.2
⌃ [187b0558] ConstructionBase v1.5.2
  [d38c429a] Contour v0.6.2
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.15.0
⌃ [a93c6f00] DataFrames v1.5.0
  [864edb3b] DataStructures v0.18.14
  [e2d170a0] DataValueInterfaces v1.0.0
  [244e2a9f] DefineSingletons v0.1.2
  [8bb1440f] DelimitedFiles v1.9.1
  [b429d917] DensityInterface v0.4.0
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
  [b4f34e82] Distances v0.10.8
  [31c24e10] Distributions v0.25.98
⌃ [ced4e74d] DistributionsAD v0.6.49
  [ffbed154] DocStringExtensions v0.9.3
  [fa6b7ba4] DualNumbers v0.6.8
⌃ [366bfd00] DynamicPPL v0.23.0
  [cad2338a] EllipticalSliceSampling v1.1.0
  [4e289a0a] EnumX v1.0.4
  [460bff9d] ExceptionUnwrapping v0.1.9
  [e2ba6199] ExprTools v0.1.9
  [c87230d0] FFMPEG v0.4.1
  [7a1cc6ca] FFTW v1.7.1
  [5789e2e9] FileIO v1.16.1
  [48062228] FilePathsBase v0.9.20
⌅ [1a297f60] FillArrays v1.0.0
  [53c48c17] FixedPointNumbers v0.8.4
  [59287772] Formatting v0.4.2
  [f6369f11] ForwardDiff v0.10.35
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
⌃ [d9f16b24] Functors v0.4.4
  [46192b85] GPUArraysCore v0.1.5
⌃ [28b8d3ca] GR v0.72.7
  [42e2da0e] Grisu v1.0.2
⌃ [cd3eb016] HTTP v1.9.8
⌃ [34004b35] HypergeometricFunctions v0.3.18
  [22cec73e] InitialValues v0.3.1
  [842dd82b] InlineStrings v1.4.0
  [505f98c9] InplaceOps v0.3.0
  [a98d9a8b] Interpolations v0.14.7
  [8197267c] IntervalSets v0.7.4
⌃ [3587e190] InverseFunctions v0.1.10
  [41ab1584] InvertedIndices v1.3.0
  [92d709cd] IrrationalConstants v0.2.2
  [c8e1da08] IterTools v1.8.0
  [82899510] IteratorInterfaceExtensions v1.0.0
  [1019f520] JLFzf v0.1.5
  [692b3bcd] JLLWrappers v1.4.1
  [682c06a0] JSON v0.21.4
⌃ [63c18a36] KernelAbstractions v0.9.6
  [5ab0869b] KernelDensity v0.6.7
  [929cbde3] LLVM v6.1.0
  [8ac3fa9e] LRUCache v1.4.1
  [b964fa9f] LaTeXStrings v1.3.0
  [23fbe1c1] Latexify v0.16.1
  [50d2b5c4] Lazy v0.15.1
⌅ [7f8f8fb0] LearnBase v0.3.0
  [1d6d02ad] LeftChildRightSiblingTrees v0.2.0
  [6f1fad26] Libtask v0.8.6
  [6fdf6af0] LogDensityProblems v2.1.1
  [996a588d] LogDensityProblemsAD v1.5.0
  [2ab3a3ac] LogExpFunctions v0.3.24
  [e6f89c97] LoggingExtras v1.0.0
  [c7f686f2] MCMCChains v6.0.3
  [be115224] MCMCDiagnosticTools v0.3.4
⌃ [9920b226] MLDataPattern v0.5.4
  [cc2ba9b6] MLDataUtils v0.5.4
  [e80e1ace] MLJModelInterface v1.8.0
  [66a33bbf] MLLabelUtils v0.5.7
  [1914dd2f] MacroTools v0.5.10
  [dbb5928d] MappedArrays v0.4.2
  [739be429] MbedTLS v1.1.7
  [442fdcdd] Measures v0.3.2
  [128add7d] MicroCollections v0.1.4
  [e1d29d7a] Missings v1.1.0
  [78c3b35d] Mocking v0.7.7
  [6f286f6a] MultivariateStats v0.10.2
⌃ [872c559c] NNlib v0.9.1
  [77ba4419] NaNMath v1.0.2
  [86f7a689] NamedArrays v0.9.8
  [c020b1a1] NaturalSort v1.0.0
  [b8a86587] NearestNeighbors v0.4.13
  [510215fc] Observables v0.5.4
  [6fe1bfb0] OffsetArrays v1.12.10
  [4d8831e6] OpenSSL v1.4.1
  [3bd65402] Optimisers v0.2.18
  [bac558e1] OrderedCollections v1.6.0
  [90014a1f] PDMats v0.11.17
  [69de0a69] Parsers v2.7.1
  [b98c9c47] Pipe v1.3.0
  [ccf2f8ad] PlotThemes v3.1.0
  [995b91a9] PlotUtils v1.3.5
  [91a5bcdd] Plots v1.38.16
  [2dfb63ee] PooledArrays v1.4.2
  [aea7be01] PrecompileTools v1.1.2
  [21216c6a] Preferences v1.4.0
⌃ [08abe8d2] PrettyTables v2.2.4
  [33c8b6b6] ProgressLogging v0.1.4
  [92933f4c] ProgressMeter v1.7.2
  [1fd47b50] QuadGK v2.8.2
⌅ [df47a6cb] RData v0.8.3
  [ce6b1742] RDatasets v0.7.7
  [74087812] Random123 v1.6.1
  [e6cf234a] RandomNumbers v1.5.3
  [b3c3ace0] RangeArrays v0.3.2
  [c84ed2f1] Ratios v0.4.5
  [c1ae055f] RealDot v0.1.0
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
⌃ [731186ca] RecursiveArrayTools v2.38.5
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.0
  [ae029012] Requires v1.3.0
  [79098fc4] Rmath v0.7.1
  [f2b01f46] Roots v2.0.17
  [7e49a35a] RuntimeGeneratedFunctions v0.5.11
⌃ [0bca4576] SciMLBase v1.93.0
  [c0aeaf25] SciMLOperators v0.3.3
  [30f210dd] ScientificTypesBase v3.0.0
  [6c6a2e73] Scratch v1.2.0
  [91c51154] SentinelArrays v1.4.0
  [efcf1570] Setfield v1.1.1
  [992d4aef] Showoff v1.0.3
  [777ac1f9] SimpleBufferStream v1.1.0
  [ce78b400] SimpleUnPack v1.1.0
  [66db9d55] SnoopPrecompile v1.0.3
  [a2af1166] SortingAlgorithms v1.1.1
  [276daf66] SpecialFunctions v2.3.0
  [171d559e] SplittablesBase v0.1.15
⌃ [90137ffa] StaticArrays v1.6.0
⌃ [1e83bf80] StaticArraysCore v1.4.0
  [64bff920] StatisticalTraits v3.2.0
  [82ae8749] StatsAPI v1.6.0
⌅ [2913bbd2] StatsBase v0.33.21
  [4c63d2b9] StatsFuns v1.3.0
⌃ [f3b207a7] StatsPlots v0.15.5
  [892a3eda] StringManipulation v0.3.0
  [09ab397b] StructArrays v0.6.15
  [2efcf032] SymbolicIndexingInterface v0.2.2
  [ab02a1b2] TableOperations v1.2.0
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.10.1
  [62fd8b95] TensorCore v0.1.1
  [5d786b92] TerminalLoggers v0.1.7
  [f269a46b] TimeZones v1.10.0
  [9f7883ad] Tracker v0.2.26
  [3bb67fe8] TranscodingStreams v0.9.13
  [28d57a85] Transducers v0.4.77
  [410a4b4d] Tricks v0.1.7
  [781d530d] TruncatedStacktraces v1.4.0
⌃ [fce5fe82] Turing v0.26.2
  [5c2747f8] URIs v1.4.2
  [1cfade01] UnicodeFun v0.4.1
  [1986cc42] Unitful v1.15.0
  [45397f5d] UnitfulLatexify v1.6.3
  [013be700] UnsafeAtomics v0.2.1
  [d80eeb9a] UnsafeAtomicsLLVM v0.1.3
  [41fe7b60] Unzip v0.2.0
  [ea10d353] WeakRefStrings v1.4.2
  [cc8bc4a8] Widgets v0.6.6
  [efce3f68] WoodburyMatrices v0.5.5
  [76eceee3] WorkerUtilities v1.6.1
  [700de1a5] ZygoteRules v0.2.3
⌅ [68821587] Arpack_jll v3.5.1+1
  [6e34b625] Bzip2_jll v1.0.8+0
  [83423d85] Cairo_jll v1.16.1+1
  [2e619515] Expat_jll v2.5.0+0
  [b22a6f82] FFMPEG_jll v4.4.2+2
  [f5851436] FFTW_jll v3.3.10+0
  [a3f928ae] Fontconfig_jll v2.13.93+0
  [d7e528f0] FreeType2_jll v2.13.1+0
  [559328eb] FriBidi_jll v1.0.10+0
  [0656b61e] GLFW_jll v3.3.8+0
⌅ [d2c73de3] GR_jll v0.72.7+0
  [78b55507] Gettext_jll v0.21.0+0
  [7746bdde] Glib_jll v2.74.0+2
  [3b182d85] Graphite2_jll v1.3.14+0
  [2e76f6c2] HarfBuzz_jll v2.8.1+1
  [1d5cc7b8] IntelOpenMP_jll v2023.1.0+0
  [aacddb02] JpegTurbo_jll v2.1.91+0
  [c1c5ebd0] LAME_jll v3.100.1+0
  [88015f11] LERC_jll v3.0.0+1
  [dad2f222] LLVMExtra_jll v0.0.23+0
  [1d63c593] LLVMOpenMP_jll v15.0.4+0
  [dd4b983a] LZO_jll v2.10.1+0
⌅ [e9f186c6] Libffi_jll v3.2.2+1
  [d4300ac3] Libgcrypt_jll v1.8.7+0
  [7e76a0d4] Libglvnd_jll v1.6.0+0
  [7add5ba3] Libgpg_error_jll v1.42.0+0
  [94ce4f54] Libiconv_jll v1.16.1+2
  [4b2f31a3] Libmount_jll v2.35.0+0
⌅ [89763e89] Libtiff_jll v4.4.0+0
  [38a345b3] Libuuid_jll v2.36.0+0
  [856f044c] MKL_jll v2023.1.0+0
  [e7412a2a] Ogg_jll v1.3.5+1
⌅ [458c3c95] OpenSSL_jll v1.1.21+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.2+0
  [30392449] Pixman_jll v0.42.2+0
  [ea2cea3b] Qt5Base_jll v5.15.3+2
  [f50d1b31] Rmath_jll v0.4.0+0
  [a2964d1f] Wayland_jll v1.21.0+0
  [2381bf8a] Wayland_protocols_jll v1.25.0+0
  [02c8fc9c] XML2_jll v2.10.3+0
  [aed1982a] XSLT_jll v1.1.34+0
  [4f6342f7] Xorg_libX11_jll v1.8.6+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.11+0
  [935fb764] Xorg_libXcursor_jll v1.2.0+4
  [a3789734] Xorg_libXdmcp_jll v1.1.4+0
  [1082639a] Xorg_libXext_jll v1.3.4+4
  [d091e8ba] Xorg_libXfixes_jll v5.0.3+4
  [a51aa0fd] Xorg_libXi_jll v1.7.10+4
  [d1454406] Xorg_libXinerama_jll v1.1.4+4
  [ec84b674] Xorg_libXrandr_jll v1.5.2+4
  [ea2f1a96] Xorg_libXrender_jll v0.9.10+4
  [14d82f49] Xorg_libpthread_stubs_jll v0.1.1+0
  [c7cfdc94] Xorg_libxcb_jll v1.15.0+0
  [cc61e674] Xorg_libxkbfile_jll v1.1.2+0
  [12413925] Xorg_xcb_util_image_jll v0.4.0+1
  [2def613f] Xorg_xcb_util_jll v0.4.0+1
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.0+1
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.9+1
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.1+1
  [35661453] Xorg_xkbcomp_jll v1.4.6+0
  [33bec58e] Xorg_xkeyboard_config_jll v2.39.0+0
  [c5fb5394] Xorg_xtrans_jll v1.5.0+0
  [3161d3a3] Zstd_jll v1.5.5+0
⌅ [214eeab7] fzf_jll v0.29.0+0
  [a4ae2306] libaom_jll v3.4.0+0
  [0ac62f75] libass_jll v0.15.1+0
  [f638f0a6] libfdk_aac_jll v2.0.2+0
  [b53b4c65] libpng_jll v1.6.38+0
  [f27f6e37] libvorbis_jll v1.3.7+1
  [1270edf5] x264_jll v2021.5.5+0
  [dfaa095f] x265_jll v3.5.0+0
  [d8fb68d0] xkbcommon_jll v1.4.1+0
  [0dad84c5] ArgTools v1.1.1
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [ade2ca70] Dates
  [8ba89e20] Distributed
  [f43a241f] Downloads v1.6.0
  [7b1f6079] FileWatching
  [9fa8497b] Future
  [b77e0a4c] InteractiveUtils
  [4af54fe1] LazyArtifacts
  [b27032c2] LibCURL v0.6.3
  [76f85450] LibGit2
  [8f399da3] Libdl
  [37e2e46d] LinearAlgebra
  [56ddb016] Logging
  [d6f4376e] Markdown
  [a63ad114] Mmap
  [ca575930] NetworkOptions v1.2.0
  [44cfe95a] Pkg v1.9.2
  [de0858da] Printf
  [3fa0cd96] REPL
  [9a3f8284] Random
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization
  [1a1011a3] SharedArrays
  [6462fe0b] Sockets
  [2f01184e] SparseArrays
  [10745b16] Statistics v1.9.0
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll v1.0.5+0
  [deac9b47] LibCURL_jll v7.84.0+0
  [29816b5a] LibSSH2_jll v1.10.2+0
  [c8ffd9c3] MbedTLS_jll v2.28.2+0
  [14a3606d] MozillaCACerts_jll v2022.10.11
  [4536629a] OpenBLAS_jll v0.3.21+4
  [05823500] OpenLibm_jll v0.8.1+0
  [efcefdf7] PCRE2_jll v10.42.0+0
  [bea87d4a] SuiteSparse_jll v5.10.1+6
  [83775a58] Zlib_jll v1.2.13+0
  [8e850b90] libblastrampoline_jll v5.8.0+0
  [8e850ede] nghttp2_jll v1.48.0+0
  [3f19e933] p7zip_jll v17.4.0+0
Info Packages marked with ⌃ and ⌅ have new versions available, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

