---
redirect_from: "tutorials/8-multinomiallogisticregression/"
title: "Bayesian Multinomial Logistic Regression"
permalink: "/:collection/:name/"
---


[Multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression) is an extension of logistic regression. Logistic regression is used to model problems in which there are exactly two possible discrete outcomes. Multinomial logistic regression is used to model problems in which there are two or more possible discrete outcomes.

In our example, we'll be using the iris dataset. The goal of the iris multiclass problem is to predict the species of a flower given measurements (in centimeters) of sepal length and width and petal length and width. There are three possible species: Iris setosa, Iris versicolor, and Iris virginica.

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

# Set a seed for reproducibility.
using Random
Random.seed!(0)

# Hide the progress prompt while sampling.
Turing.setprogress!(false);
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
   1 │         5.9         3.2          4.8         1.8  versicolor
   2 │         7.7         3.8          6.7         2.2  virginica
   3 │         5.9         3.0          4.2         1.5  versicolor
   4 │         6.3         2.9          5.6         1.8  virginica
   5 │         6.3         2.7          4.9         1.8  virginica
   6 │         4.8         3.4          1.6         0.2  setosa
   7 │         6.5         3.0          5.5         1.8  virginica
   8 │         5.0         3.4          1.5         0.2  setosa
  ⋮  │      ⋮           ⋮            ⋮           ⋮           ⋮
  14 │         5.6         2.8          4.9         2.0  virginica
  15 │         6.4         2.8          5.6         2.1  virginica
  16 │         7.2         3.6          6.1         2.5  virginica
  17 │         5.6         2.5          3.9         1.1  versicolor
  18 │         5.8         2.8          5.1         2.4  virginica
  19 │         5.8         2.7          4.1         1.0  versicolor
  20 │         5.0         3.4          1.5         0.2  setosa
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
   1 │ setosa      1
   2 │ virginica   3
   3 │ setosa      1
   4 │ versicolor  2
   5 │ setosa      1
   6 │ versicolor  2
   7 │ versicolor  2
   8 │ setosa      1
  ⋮  │     ⋮             ⋮
  14 │ setosa      1
  15 │ virginica   3
  16 │ virginica   3
  17 │ setosa      1
  18 │ versicolor  2
  19 │ virginica   3
  20 │ versicolor  2
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
    coefficients_versicolor ~ MvNormal(4, σ)
    coefficients_virginica ~ MvNormal(4, σ)

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

Now we can run our sampler. This time we'll use [`HMC`](http://turing.ml/docs/library/#Turing.HMC) to sample from our posterior.

```julia
m = logistic_regression(train_features, train_target, 1)
chain = sample(m, HMC(0.05, 10), MCMCThreads(), 1_500, 3)
```

```
Error: TaskFailedException

    nested task error: TaskFailedException
    Stacktrace:
     [1] wait
       @ ./task.jl:322 [inlined]
     [2] threading_run(func::Function)
       @ Base.Threads ./threadingconstructs.jl:34
     [3] macro expansion
       @ ./threadingconstructs.jl:93 [inlined]
     [4] macro expansion
       @ /cache/julia-buildkite-plugin/depots/7aa0085e-79a4-45f3-a5bd-9743c
91cf3da/packages/AbstractMCMC/BPJCW/src/sample.jl:342 [inlined]
     [5] (::AbstractMCMC.var"#36#48"{Bool, Base.Iterators.Pairs{Symbol, Uni
onAll, Tuple{Symbol}, NamedTuple{(:chain_type,), Tuple{UnionAll}}}, Int64, 
Int64, Vector{Any}, Vector{UInt64}, Vector{DynamicPPL.Sampler{Turing.Infere
nce.HMC{Turing.Core.ForwardDiffAD{40}, (), AdvancedHMC.UnitEuclideanMetric}
}}, Vector{DynamicPPL.Model{Main.##WeaveSandBox#289.var"#1#2", (:x, :y, :σ)
, (), (), Tuple{Matrix{Float64}, SubArray{Union{Nothing, Int64}, 1, Vector{
Union{Nothing, Int64}}, Tuple{Vector{Int64}}, false}, Int64}, Tuple{}}}, Ve
ctor{Random._GLOBAL_RNG}})()
       @ AbstractMCMC ./task.jl:411
    
        nested task error: MethodError: no method matching Int64(::Irration
al{:log2π})
        Closest candidates are:
          (::Type{T})(::T) where T<:Number at boot.jl:760
          (::Type{T})(!Matched::AbstractChar) where T<:Union{AbstractChar, 
Number} at char.jl:50
          (::Type{T})(!Matched::BigInt) where T<:Union{Int128, Int16, Int32
, Int64, Int8} at gmp.jl:356
          ...
        Stacktrace:
          [1] convert(#unused#::Type{Int64}, x::Irrational{:log2π})
            @ Base ./number.jl:7
          [2] mvnormal_c0(g::Distributions.MvNormal{Int64, PDMats.ScalMat{I
nt64}, FillArrays.Zeros{Int64, 1, Tuple{Base.OneTo{Int64}}}})
            @ Distributions /cache/julia-buildkite-plugin/depots/7aa0085e-7
9a4-45f3-a5bd-9743c91cf3da/packages/Distributions/t65ji/src/multivariate/mv
normal.jl:99
          [3] _logpdf(d::Distributions.MvNormal{Int64, PDMats.ScalMat{Int64
}, FillArrays.Zeros{Int64, 1, Tuple{Base.OneTo{Int64}}}}, x::Vector{Float64
})
            @ Distributions /cache/julia-buildkite-plugin/depots/7aa0085e-7
9a4-45f3-a5bd-9743c91cf3da/packages/Distributions/t65ji/src/multivariate/mv
normal.jl:127
          [4] logpdf(d::Distributions.MvNormal{Int64, PDMats.ScalMat{Int64}
, FillArrays.Zeros{Int64, 1, Tuple{Base.OneTo{Int64}}}}, X::Vector{Float64}
)
            @ Distributions /cache/julia-buildkite-plugin/depots/7aa0085e-7
9a4-45f3-a5bd-9743c91cf3da/packages/Distributions/t65ji/src/multivariates.j
l:201
          [5] logpdf_with_trans(d::Distributions.MvNormal{Int64, PDMats.Sca
lMat{Int64}, FillArrays.Zeros{Int64, 1, Tuple{Base.OneTo{Int64}}}}, x::Vect
or{Float64}, transform::Bool)
            @ Bijectors /cache/julia-buildkite-plugin/depots/7aa0085e-79a4-
45f3-a5bd-9743c91cf3da/packages/Bijectors/LmARY/src/Bijectors.jl:129
          [6] assume(rng::Random._GLOBAL_RNG, sampler::DynamicPPL.SampleFro
mUniform, dist::Distributions.MvNormal{Int64, PDMats.ScalMat{Int64}, FillAr
rays.Zeros{Int64, 1, Tuple{Base.OneTo{Int64}}}}, vn::AbstractPPL.VarName{:c
oefficients_versicolor, Tuple{}}, vi::DynamicPPL.UntypedVarInfo{DynamicPPL.
Metadata{Dict{AbstractPPL.VarName, Int64}, Vector{Distributions.Distributio
n}, Vector{AbstractPPL.VarName}, Vector{Real}, Vector{Set{DynamicPPL.Select
or}}}, Float64})
            @ DynamicPPL /cache/julia-buildkite-plugin/depots/7aa0085e-79a4
-45f3-a5bd-9743c91cf3da/packages/DynamicPPL/F7F1M/src/context_implementatio
ns.jl:262
          [7] tilde_assume(rng::Random._GLOBAL_RNG, #unused#::DynamicPPL.De
faultContext, sampler::DynamicPPL.SampleFromUniform, right::Distributions.M
vNormal{Int64, PDMats.ScalMat{Int64}, FillArrays.Zeros{Int64, 1, Tuple{Base
.OneTo{Int64}}}}, vn::AbstractPPL.VarName{:coefficients_versicolor, Tuple{}
}, inds::Tuple{}, vi::DynamicPPL.UntypedVarInfo{DynamicPPL.Metadata{Dict{Ab
stractPPL.VarName, Int64}, Vector{Distributions.Distribution}, Vector{Abstr
actPPL.VarName}, Vector{Real}, Vector{Set{DynamicPPL.Selector}}}, Float64})
            @ DynamicPPL /cache/julia-buildkite-plugin/depots/7aa0085e-79a4
-45f3-a5bd-9743c91cf3da/packages/DynamicPPL/F7F1M/src/context_implementatio
ns.jl:42
          [8] tilde_assume(context::DynamicPPL.SamplingContext{DynamicPPL.S
ampleFromUniform, DynamicPPL.DefaultContext, Random._GLOBAL_RNG}, right::Di
stributions.MvNormal{Int64, PDMats.ScalMat{Int64}, FillArrays.Zeros{Int64, 
1, Tuple{Base.OneTo{Int64}}}}, vn::AbstractPPL.VarName{:coefficients_versic
olor, Tuple{}}, inds::Tuple{}, vi::DynamicPPL.UntypedVarInfo{DynamicPPL.Met
adata{Dict{AbstractPPL.VarName, Int64}, Vector{Distributions.Distribution},
 Vector{AbstractPPL.VarName}, Vector{Real}, Vector{Set{DynamicPPL.Selector}
}}, Float64})
            @ DynamicPPL /cache/julia-buildkite-plugin/depots/7aa0085e-79a4
-45f3-a5bd-9743c91cf3da/packages/DynamicPPL/F7F1M/src/context_implementatio
ns.jl:34
          [9] tilde_assume!(context::DynamicPPL.SamplingContext{DynamicPPL.
SampleFromUniform, DynamicPPL.DefaultContext, Random._GLOBAL_RNG}, right::D
istributions.MvNormal{Int64, PDMats.ScalMat{Int64}, FillArrays.Zeros{Int64,
 1, Tuple{Base.OneTo{Int64}}}}, vn::AbstractPPL.VarName{:coefficients_versi
color, Tuple{}}, inds::Tuple{}, vi::DynamicPPL.UntypedVarInfo{DynamicPPL.Me
tadata{Dict{AbstractPPL.VarName, Int64}, Vector{Distributions.Distribution}
, Vector{AbstractPPL.VarName}, Vector{Real}, Vector{Set{DynamicPPL.Selector
}}}, Float64})
            @ DynamicPPL /cache/julia-buildkite-plugin/depots/7aa0085e-79a4
-45f3-a5bd-9743c91cf3da/packages/DynamicPPL/F7F1M/src/context_implementatio
ns.jl:131
         [10] #1
            @ /cache/build/exclusive-amdci3-0/julialang/turingtutorials/tut
orials/08-multinomial-logistic-regression/08_multinomial-logistic-regressio
n.jmd:11 [inlined]
         [11] (::Main.##WeaveSandBox#289.var"#1#2")(__model__::DynamicPPL.M
odel{Main.##WeaveSandBox#289.var"#1#2", (:x, :y, :σ), (), (), Tuple{Matrix{
Float64}, SubArray{Union{Nothing, Int64}, 1, Vector{Union{Nothing, Int64}},
 Tuple{Vector{Int64}}, false}, Int64}, Tuple{}}, __varinfo__::DynamicPPL.Un
typedVarInfo{DynamicPPL.Metadata{Dict{AbstractPPL.VarName, Int64}, Vector{D
istributions.Distribution}, Vector{AbstractPPL.VarName}, Vector{Real}, Vect
or{Set{DynamicPPL.Selector}}}, Float64}, __context__::DynamicPPL.SamplingCo
ntext{DynamicPPL.SampleFromUniform, DynamicPPL.DefaultContext, Random._GLOB
AL_RNG}, x::Matrix{Float64}, y::SubArray{Union{Nothing, Int64}, 1, Vector{U
nion{Nothing, Int64}}, Tuple{Vector{Int64}}, false}, σ::Int64)
            @ Main.##WeaveSandBox#289 ./none:0
         [12] macro expansion
            @ /cache/julia-buildkite-plugin/depots/7aa0085e-79a4-45f3-a5bd-
9743c91cf3da/packages/DynamicPPL/F7F1M/src/model.jl:0 [inlined]
         [13] _evaluate(model::DynamicPPL.Model{Main.##WeaveSandBox#289.var
"#1#2", (:x, :y, :σ), (), (), Tuple{Matrix{Float64}, SubArray{Union{Nothing
, Int64}, 1, Vector{Union{Nothing, Int64}}, Tuple{Vector{Int64}}, false}, I
nt64}, Tuple{}}, varinfo::DynamicPPL.UntypedVarInfo{DynamicPPL.Metadata{Dic
t{AbstractPPL.VarName, Int64}, Vector{Distributions.Distribution}, Vector{A
bstractPPL.VarName}, Vector{Real}, Vector{Set{DynamicPPL.Selector}}}, Float
64}, context::DynamicPPL.SamplingContext{DynamicPPL.SampleFromUniform, Dyna
micPPL.DefaultContext, Random._GLOBAL_RNG})
            @ DynamicPPL /cache/julia-buildkite-plugin/depots/7aa0085e-79a4
-45f3-a5bd-9743c91cf3da/packages/DynamicPPL/F7F1M/src/model.jl:156
         [14] evaluate_threadunsafe(model::DynamicPPL.Model{Main.##WeaveSan
dBox#289.var"#1#2", (:x, :y, :σ), (), (), Tuple{Matrix{Float64}, SubArray{U
nion{Nothing, Int64}, 1, Vector{Union{Nothing, Int64}}, Tuple{Vector{Int64}
}, false}, Int64}, Tuple{}}, varinfo::DynamicPPL.UntypedVarInfo{DynamicPPL.
Metadata{Dict{AbstractPPL.VarName, Int64}, Vector{Distributions.Distributio
n}, Vector{AbstractPPL.VarName}, Vector{Real}, Vector{Set{DynamicPPL.Select
or}}}, Float64}, context::DynamicPPL.SamplingContext{DynamicPPL.SampleFromU
niform, DynamicPPL.DefaultContext, Random._GLOBAL_RNG})
            @ DynamicPPL /cache/julia-buildkite-plugin/depots/7aa0085e-79a4
-45f3-a5bd-9743c91cf3da/packages/DynamicPPL/F7F1M/src/model.jl:129
         [15] (::DynamicPPL.Model{Main.##WeaveSandBox#289.var"#1#2", (:x, :
y, :σ), (), (), Tuple{Matrix{Float64}, SubArray{Union{Nothing, Int64}, 1, V
ector{Union{Nothing, Int64}}, Tuple{Vector{Int64}}, false}, Int64}, Tuple{}
})(varinfo::DynamicPPL.UntypedVarInfo{DynamicPPL.Metadata{Dict{AbstractPPL.
VarName, Int64}, Vector{Distributions.Distribution}, Vector{AbstractPPL.Var
Name}, Vector{Real}, Vector{Set{DynamicPPL.Selector}}}, Float64}, context::
DynamicPPL.SamplingContext{DynamicPPL.SampleFromUniform, DynamicPPL.Default
Context, Random._GLOBAL_RNG})
            @ DynamicPPL /cache/julia-buildkite-plugin/depots/7aa0085e-79a4
-45f3-a5bd-9743c91cf3da/packages/DynamicPPL/F7F1M/src/model.jl:97
         [16] (::DynamicPPL.Model{Main.##WeaveSandBox#289.var"#1#2", (:x, :
y, :σ), (), (), Tuple{Matrix{Float64}, SubArray{Union{Nothing, Int64}, 1, V
ector{Union{Nothing, Int64}}, Tuple{Vector{Int64}}, false}, Int64}, Tuple{}
})(rng::Random._GLOBAL_RNG, varinfo::DynamicPPL.UntypedVarInfo{DynamicPPL.M
etadata{Dict{AbstractPPL.VarName, Int64}, Vector{Distributions.Distribution
}, Vector{AbstractPPL.VarName}, Vector{Real}, Vector{Set{DynamicPPL.Selecto
r}}}, Float64}, sampler::DynamicPPL.SampleFromUniform, context::DynamicPPL.
DefaultContext)
            @ DynamicPPL /cache/julia-buildkite-plugin/depots/7aa0085e-79a4
-45f3-a5bd-9743c91cf3da/packages/DynamicPPL/F7F1M/src/model.jl:91
         [17] DynamicPPL.VarInfo(rng::Random._GLOBAL_RNG, model::DynamicPPL
.Model{Main.##WeaveSandBox#289.var"#1#2", (:x, :y, :σ), (), (), Tuple{Matri
x{Float64}, SubArray{Union{Nothing, Int64}, 1, Vector{Union{Nothing, Int64}
}, Tuple{Vector{Int64}}, false}, Int64}, Tuple{}}, sampler::DynamicPPL.Samp
leFromUniform, context::DynamicPPL.DefaultContext)
            @ DynamicPPL /cache/julia-buildkite-plugin/depots/7aa0085e-79a4
-45f3-a5bd-9743c91cf3da/packages/DynamicPPL/F7F1M/src/varinfo.jl:132
         [18] DynamicPPL.VarInfo(rng::Random._GLOBAL_RNG, model::DynamicPPL
.Model{Main.##WeaveSandBox#289.var"#1#2", (:x, :y, :σ), (), (), Tuple{Matri
x{Float64}, SubArray{Union{Nothing, Int64}, 1, Vector{Union{Nothing, Int64}
}, Tuple{Vector{Int64}}, false}, Int64}, Tuple{}}, sampler::DynamicPPL.Samp
leFromUniform)
            @ DynamicPPL /cache/julia-buildkite-plugin/depots/7aa0085e-79a4
-45f3-a5bd-9743c91cf3da/packages/DynamicPPL/F7F1M/src/varinfo.jl:131
         [19] step(rng::Random._GLOBAL_RNG, model::DynamicPPL.Model{Main.##
WeaveSandBox#289.var"#1#2", (:x, :y, :σ), (), (), Tuple{Matrix{Float64}, Su
bArray{Union{Nothing, Int64}, 1, Vector{Union{Nothing, Int64}}, Tuple{Vecto
r{Int64}}, false}, Int64}, Tuple{}}, spl::DynamicPPL.Sampler{Turing.Inferen
ce.HMC{Turing.Core.ForwardDiffAD{40}, (), AdvancedHMC.UnitEuclideanMetric}}
; resume_from::Nothing, kwargs::Base.Iterators.Pairs{Union{}, Union{}, Tupl
e{}, NamedTuple{(), Tuple{}}})
            @ DynamicPPL /cache/julia-buildkite-plugin/depots/7aa0085e-79a4
-45f3-a5bd-9743c91cf3da/packages/DynamicPPL/F7F1M/src/sampler.jl:69
         [20] step(rng::Random._GLOBAL_RNG, model::DynamicPPL.Model{Main.##
WeaveSandBox#289.var"#1#2", (:x, :y, :σ), (), (), Tuple{Matrix{Float64}, Su
bArray{Union{Nothing, Int64}, 1, Vector{Union{Nothing, Int64}}, Tuple{Vecto
r{Int64}}, false}, Int64}, Tuple{}}, spl::DynamicPPL.Sampler{Turing.Inferen
ce.HMC{Turing.Core.ForwardDiffAD{40}, (), AdvancedHMC.UnitEuclideanMetric}}
)
            @ DynamicPPL /cache/julia-buildkite-plugin/depots/7aa0085e-79a4
-45f3-a5bd-9743c91cf3da/packages/DynamicPPL/F7F1M/src/sampler.jl:62
         [21] macro expansion
            @ /cache/julia-buildkite-plugin/depots/7aa0085e-79a4-45f3-a5bd-
9743c91cf3da/packages/AbstractMCMC/BPJCW/src/sample.jl:123 [inlined]
         [22] macro expansion
            @ /cache/julia-buildkite-plugin/depots/7aa0085e-79a4-45f3-a5bd-
9743c91cf3da/packages/AbstractMCMC/BPJCW/src/logging.jl:15 [inlined]
         [23] mcmcsample(rng::Random._GLOBAL_RNG, model::DynamicPPL.Model{M
ain.##WeaveSandBox#289.var"#1#2", (:x, :y, :σ), (), (), Tuple{Matrix{Float6
4}, SubArray{Union{Nothing, Int64}, 1, Vector{Union{Nothing, Int64}}, Tuple
{Vector{Int64}}, false}, Int64}, Tuple{}}, sampler::DynamicPPL.Sampler{Turi
ng.Inference.HMC{Turing.Core.ForwardDiffAD{40}, (), AdvancedHMC.UnitEuclide
anMetric}}, N::Int64; progress::Bool, progressname::String, callback::Nothi
ng, discard_initial::Int64, thinning::Int64, chain_type::Type, kwargs::Base
.Iterators.Pairs{Union{}, Union{}, Tuple{}, NamedTuple{(), Tuple{}}})
            @ AbstractMCMC /cache/julia-buildkite-plugin/depots/7aa0085e-79
a4-45f3-a5bd-9743c91cf3da/packages/AbstractMCMC/BPJCW/src/sample.jl:114
         [24] sample(rng::Random._GLOBAL_RNG, model::DynamicPPL.Model{Main.
##WeaveSandBox#289.var"#1#2", (:x, :y, :σ), (), (), Tuple{Matrix{Float64}, 
SubArray{Union{Nothing, Int64}, 1, Vector{Union{Nothing, Int64}}, Tuple{Vec
tor{Int64}}, false}, Int64}, Tuple{}}, sampler::DynamicPPL.Sampler{Turing.I
nference.HMC{Turing.Core.ForwardDiffAD{40}, (), AdvancedHMC.UnitEuclideanMe
tric}}, N::Int64; chain_type::Type, resume_from::Nothing, progress::Bool, k
wargs::Base.Iterators.Pairs{Union{}, Union{}, Tuple{}, NamedTuple{(), Tuple
{}}})
            @ Turing.Inference /cache/julia-buildkite-plugin/depots/7aa0085
e-79a4-45f3-a5bd-9743c91cf3da/packages/Turing/YGtAo/src/inference/Inference
.jl:156
         [25] macro expansion
            @ /cache/julia-buildkite-plugin/depots/7aa0085e-79a4-45f3-a5bd-
9743c91cf3da/packages/AbstractMCMC/BPJCW/src/sample.jl:351 [inlined]
         [26] (::AbstractMCMC.var"#1082#threadsfor_fun#49"{UnitRange{Int64}
, Bool, Base.Iterators.Pairs{Symbol, UnionAll, Tuple{Symbol}, NamedTuple{(:
chain_type,), Tuple{UnionAll}}}, Int64, Vector{Any}, Vector{UInt64}, Vector
{DynamicPPL.Sampler{Turing.Inference.HMC{Turing.Core.ForwardDiffAD{40}, (),
 AdvancedHMC.UnitEuclideanMetric}}}, Vector{DynamicPPL.Model{Main.##WeaveSa
ndBox#289.var"#1#2", (:x, :y, :σ), (), (), Tuple{Matrix{Float64}, SubArray{
Union{Nothing, Int64}, 1, Vector{Union{Nothing, Int64}}, Tuple{Vector{Int64
}}, false}, Int64}, Tuple{}}}, Vector{Random._GLOBAL_RNG}})(onethread::Bool
)
            @ AbstractMCMC ./threadingconstructs.jl:81
         [27] (::AbstractMCMC.var"#1082#threadsfor_fun#49"{UnitRange{Int64}
, Bool, Base.Iterators.Pairs{Symbol, UnionAll, Tuple{Symbol}, NamedTuple{(:
chain_type,), Tuple{UnionAll}}}, Int64, Vector{Any}, Vector{UInt64}, Vector
{DynamicPPL.Sampler{Turing.Inference.HMC{Turing.Core.ForwardDiffAD{40}, (),
 AdvancedHMC.UnitEuclideanMetric}}}, Vector{DynamicPPL.Model{Main.##WeaveSa
ndBox#289.var"#1#2", (:x, :y, :σ), (), (), Tuple{Matrix{Float64}, SubArray{
Union{Nothing, Int64}, 1, Vector{Union{Nothing, Int64}}, Tuple{Vector{Int64
}}, false}, Int64}, Tuple{}}}, Vector{Random._GLOBAL_RNG}})()
            @ AbstractMCMC ./threadingconstructs.jl:48
```





Since we ran multiple chains, we may as well do a spot check to make sure each chain converges around similar points.

```julia
plot(chain)
```

```
Error: UndefVarError: chain not defined
```





Looks good!

We can also use the `corner` function from MCMCChains to show the distributions of the various parameters of our multinomial logistic regression. The corner function requires MCMCChains and StatsPlots.

```julia
corner(
    chain,
    MCMCChains.namesingroup(chain, :coefficients_versicolor);
    label=[string(i) for i in 1:4],
)
```

```
Error: UndefVarError: chain not defined
```



```julia
corner(
    chain,
    MCMCChains.namesingroup(chain, :coefficients_virginica);
    label=[string(i) for i in 1:4],
)
```

```
Error: UndefVarError: chain not defined
```





Fortunately the corner plots appear to demonstrate unimodal distributions for each of our parameters, so it should be straightforward to take the means of each parameter's sampled values to estimate our model to make predictions.

## Making Predictions

How do we test how well the model actually predicts whether someone is likely to default? We need to build a `prediction` function that takes the test dataset and runs it through the average parameter calculated during sampling.

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
Error: UndefVarError: chain not defined
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
Number of `setosa`: 24
Error: UndefVarError: predictions not defined
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
Julia Version 1.6.5
Commit 9058264a69 (2021-12-19 12:30 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, znver2)
Environment:
  BUILDKITE_PLUGIN_JULIA_CACHE_DIR = /cache/julia-buildkite-plugin
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/7aa0085e-79a4-45f3-a5bd-9743c91cf3da

```

Package Information:

```
      Status `/cache/build/exclusive-amdci3-0/julialang/turingtutorials/tutorials/08-multinomial-logistic-regression/Project.toml`
  [a93c6f00] DataFrames v1.3.2
  [b4f34e82] Distances v0.10.7
  [31c24e10] Distributions v0.25.14
  [38e38edf] GLM v1.6.1
  [c7f686f2] MCMCChains v4.14.1
  [cc2ba9b6] MLDataUtils v0.5.4
  [872c559c] NNlib v0.7.34
  [91a5bcdd] Plots v1.25.5
  [ce6b1742] RDatasets v0.7.7
  [4c63d2b9] StatsFuns v0.9.9
  [f3b207a7] StatsPlots v0.14.30
  [fce5fe82] Turing v0.16.6
  [9a3f8284] Random
```

And the full manifest:

```
      Status `/cache/build/exclusive-amdci3-0/julialang/turingtutorials/tutorials/08-multinomial-logistic-regression/Manifest.toml`
  [621f4979] AbstractFFTs v1.0.1
  [80f14c24] AbstractMCMC v3.2.1
  [7a57a42e] AbstractPPL v0.1.4
  [1520ce14] AbstractTrees v0.3.4
  [79e6a3ab] Adapt v3.3.3
  [0bf59076] AdvancedHMC v0.3.3
  [5b7e9947] AdvancedMH v0.6.6
  [576499cb] AdvancedPS v0.2.4
  [b5ca4192] AdvancedVI v0.1.3
  [dce04be8] ArgCheck v2.3.0
  [7d9fca2a] Arpack v0.4.0
  [4fba245c] ArrayInterface v4.0.3
  [13072b0f] AxisAlgorithms v1.0.1
  [39de3d68] AxisArrays v0.4.4
  [198e06fe] BangBang v0.3.35
  [9718e550] Baselet v0.1.1
  [76274a88] Bijectors v0.9.7
  [336ed68f] CSV v0.10.2
  [324d7699] CategoricalArrays v0.10.2
  [082447d4] ChainRules v0.8.25
  [d360d2e6] ChainRulesCore v0.10.13
  [aaaa29a8] Clustering v0.14.2
  [944b1d66] CodecZlib v0.7.0
  [35d6a980] ColorSchemes v3.17.1
  [3da002f7] ColorTypes v0.11.0
  [5ae59095] Colors v0.12.8
  [861a8166] Combinatorics v1.0.2
  [38540f10] CommonSolve v0.2.0
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v3.41.0
  [a33af91c] CompositionsBase v0.1.1
  [88cd18e8] ConsoleProgressMonitor v0.1.2
  [187b0558] ConstructionBase v1.3.0
  [d38c429a] Contour v0.5.7
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.9.0
  [a93c6f00] DataFrames v1.3.2
  [864edb3b] DataStructures v0.18.11
  [e2d170a0] DataValueInterfaces v1.0.0
  [e7dc6d0d] DataValues v0.4.13
  [244e2a9f] DefineSingletons v0.1.2
  [163ba53b] DiffResults v1.0.3
  [b552c78f] DiffRules v1.5.0
  [b4f34e82] Distances v0.10.7
  [31c24e10] Distributions v0.25.14
  [ced4e74d] DistributionsAD v0.6.29
  [ffbed154] DocStringExtensions v0.8.6
  [366bfd00] DynamicPPL v0.12.4
  [da5c29d0] EllipsisNotation v1.3.0
  [cad2338a] EllipticalSliceSampling v0.4.6
  [e2ba6199] ExprTools v0.1.8
  [c87230d0] FFMPEG v0.4.1
  [7a1cc6ca] FFTW v1.4.5
  [5789e2e9] FileIO v1.13.0
  [48062228] FilePathsBase v0.9.17
  [1a297f60] FillArrays v0.11.9
  [6a86dc24] FiniteDiff v2.10.1
  [53c48c17] FixedPointNumbers v0.8.4
  [59287772] Formatting v0.4.2
  [f6369f11] ForwardDiff v0.10.25
  [d9f16b24] Functors v0.2.8
  [38e38edf] GLM v1.6.1
  [28b8d3ca] GR v0.63.1
  [5c1252a2] GeometryBasics v0.4.1
  [42e2da0e] Grisu v1.0.2
  [cd3eb016] HTTP v0.9.17
  [615f187c] IfElse v0.1.1
  [83e8ac13] IniFile v0.5.0
  [22cec73e] InitialValues v0.3.1
  [842dd82b] InlineStrings v1.1.2
  [505f98c9] InplaceOps v0.3.0
  [a98d9a8b] Interpolations v0.13.5
  [8197267c] IntervalSets v0.5.3
  [41ab1584] InvertedIndices v1.1.0
  [92d709cd] IrrationalConstants v0.1.1
  [c8e1da08] IterTools v1.4.0
  [42fd0dbc] IterativeSolvers v0.9.2
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.4.1
  [682c06a0] JSON v0.21.3
  [5ab0869b] KernelDensity v0.6.3
  [b964fa9f] LaTeXStrings v1.3.0
  [23fbe1c1] Latexify v0.15.11
  [7f8f8fb0] LearnBase v0.3.0
  [1d6d02ad] LeftChildRightSiblingTrees v0.1.3
  [6f1fad26] Libtask v0.5.3
  [2ab3a3ac] LogExpFunctions v0.3.0
  [e6f89c97] LoggingExtras v0.4.7
  [c7f686f2] MCMCChains v4.14.1
  [9920b226] MLDataPattern v0.5.4
  [cc2ba9b6] MLDataUtils v0.5.4
  [e80e1ace] MLJModelInterface v1.3.6
  [66a33bbf] MLLabelUtils v0.5.7
  [1914dd2f] MacroTools v0.5.9
  [dbb5928d] MappedArrays v0.4.1
  [739be429] MbedTLS v1.0.3
  [442fdcdd] Measures v0.3.1
  [128add7d] MicroCollections v0.1.2
  [e1d29d7a] Missings v1.0.2
  [78c3b35d] Mocking v0.7.3
  [6f286f6a] MultivariateStats v0.8.0
  [872c559c] NNlib v0.7.34
  [77ba4419] NaNMath v0.3.7
  [86f7a689] NamedArrays v0.9.6
  [c020b1a1] NaturalSort v1.0.0
  [b8a86587] NearestNeighbors v0.4.9
  [8913a72c] NonlinearSolve v0.3.14
  [510215fc] Observables v0.4.0
  [6fe1bfb0] OffsetArrays v1.10.8
  [bac558e1] OrderedCollections v1.4.1
  [90014a1f] PDMats v0.11.5
  [69de0a69] Parsers v2.2.2
  [ccf2f8ad] PlotThemes v2.0.1
  [995b91a9] PlotUtils v1.1.3
  [91a5bcdd] Plots v1.25.5
  [2dfb63ee] PooledArrays v1.4.0
  [21216c6a] Preferences v1.2.3
  [08abe8d2] PrettyTables v1.3.1
  [33c8b6b6] ProgressLogging v0.1.4
  [92933f4c] ProgressMeter v1.7.1
  [1fd47b50] QuadGK v2.4.2
  [df47a6cb] RData v0.8.3
  [ce6b1742] RDatasets v0.7.7
  [b3c3ace0] RangeArrays v0.3.2
  [c84ed2f1] Ratios v0.4.2
  [3cdcf5f2] RecipesBase v1.2.1
  [01d81517] RecipesPipeline v0.4.1
  [731186ca] RecursiveArrayTools v2.24.2
  [f2c3362d] RecursiveFactorization v0.1.0
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v0.1.3
  [ae029012] Requires v1.3.0
  [79098fc4] Rmath v0.7.0
  [0bca4576] SciMLBase v1.26.1
  [30f210dd] ScientificTypesBase v3.0.0
  [6c6a2e73] Scratch v1.1.0
  [91c51154] SentinelArrays v1.3.12
  [efcf1570] Setfield v0.8.2
  [1277b4bf] ShiftedArrays v1.0.0
  [992d4aef] Showoff v1.0.3
  [a2af1166] SortingAlgorithms v1.0.1
  [276daf66] SpecialFunctions v1.8.3
  [171d559e] SplittablesBase v0.1.14
  [aedffcd0] Static v0.5.5
  [90137ffa] StaticArrays v1.3.5
  [64bff920] StatisticalTraits v3.0.0
  [82ae8749] StatsAPI v1.2.1
  [2913bbd2] StatsBase v0.33.16
  [4c63d2b9] StatsFuns v0.9.9
  [3eaba693] StatsModels v0.6.28
  [f3b207a7] StatsPlots v0.14.30
  [09ab397b] StructArrays v0.6.5
  [ab02a1b2] TableOperations v1.2.0
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.6.1
  [5d786b92] TerminalLoggers v0.1.5
  [f269a46b] TimeZones v1.7.1
  [9f7883ad] Tracker v0.2.19
  [3bb67fe8] TranscodingStreams v0.9.6
  [28d57a85] Transducers v0.4.72
  [a2a6695c] TreeViews v0.3.0
  [fce5fe82] Turing v0.16.6
  [5c2747f8] URIs v1.3.0
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
  [41fe7b60] Unzip v0.1.2
  [ea10d353] WeakRefStrings v1.4.1
  [cc8bc4a8] Widgets v0.6.5
  [efce3f68] WoodburyMatrices v0.5.5
  [700de1a5] ZygoteRules v0.2.2
  [68821587] Arpack_jll v3.5.0+3
  [6e34b625] Bzip2_jll v1.0.8+0
  [83423d85] Cairo_jll v1.16.1+1
  [5ae413db] EarCut_jll v2.2.3+0
  [2e619515] Expat_jll v2.4.4+0
  [b22a6f82] FFMPEG_jll v4.4.0+0
  [f5851436] FFTW_jll v3.3.10+0
  [a3f928ae] Fontconfig_jll v2.13.93+0
  [d7e528f0] FreeType2_jll v2.10.4+0
  [559328eb] FriBidi_jll v1.0.10+0
  [0656b61e] GLFW_jll v3.3.6+0
  [d2c73de3] GR_jll v0.64.0+0
  [78b55507] Gettext_jll v0.21.0+0
  [7746bdde] Glib_jll v2.68.3+2
  [3b182d85] Graphite2_jll v1.3.14+0
  [2e76f6c2] HarfBuzz_jll v2.8.1+1
  [1d5cc7b8] IntelOpenMP_jll v2018.0.3+2
  [aacddb02] JpegTurbo_jll v2.1.2+0
  [c1c5ebd0] LAME_jll v3.100.1+0
  [dd4b983a] LZO_jll v2.10.1+0
  [e9f186c6] Libffi_jll v3.2.2+1
  [d4300ac3] Libgcrypt_jll v1.8.7+0
  [7e76a0d4] Libglvnd_jll v1.3.0+3
  [7add5ba3] Libgpg_error_jll v1.42.0+0
  [94ce4f54] Libiconv_jll v1.16.1+1
  [4b2f31a3] Libmount_jll v2.35.0+0
  [3ae2931a] Libtask_jll v0.4.3+0
  [89763e89] Libtiff_jll v4.3.0+0
  [38a345b3] Libuuid_jll v2.36.0+0
  [856f044c] MKL_jll v2021.1.1+2
  [e7412a2a] Ogg_jll v1.3.5+1
  [458c3c95] OpenSSL_jll v1.1.13+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.2+0
  [2f80f16e] PCRE_jll v8.44.0+0
  [30392449] Pixman_jll v0.40.1+0
  [ea2cea3b] Qt5Base_jll v5.15.3+0
  [f50d1b31] Rmath_jll v0.3.0+0
  [a2964d1f] Wayland_jll v1.19.0+0
  [2381bf8a] Wayland_protocols_jll v1.23.0+0
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

