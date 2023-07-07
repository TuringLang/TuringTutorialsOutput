
# Import Turing.
using Turing

# Package for loading the data set.
using RDatasets

# Package for visualization.
using StatsPlots

# Functionality for splitting the data.
using MLUtils: splitobs

# Functionality for constructing arrays with identical elements efficiently.
using FillArrays

# Functionality for normalizing the data and evaluating the model predictions.
using StatsBase

# Functionality for working with scaled identity matrices.
using LinearAlgebra

# Set a seed for reproducibility.
using Random
Random.seed!(0);


# Load the dataset.
data = RDatasets.dataset("datasets", "mtcars")

# Show the first six rows of the dataset.
first(data, 6)


size(data)


# Remove the model column.
select!(data, Not(:Model))

# Split our dataset 70%/30% into training/test sets.
trainset, testset = map(DataFrame, splitobs(data; at=0.7, shuffle=true))

# Turing requires data in matrix form.
target = :MPG
train = Matrix(select(trainset, Not(target)))
test = Matrix(select(testset, Not(target)))
train_target = trainset[:, target]
test_target = testset[:, target]

# Standardize the features.
dt_features = fit(ZScoreTransform, train; dims=1)
StatsBase.transform!(dt_features, train)
StatsBase.transform!(dt_features, test)

# Standardize the targets.
dt_targets = fit(ZScoreTransform, train_target)
StatsBase.transform!(dt_targets, train_target)
StatsBase.transform!(dt_targets, test_target);


# Bayesian linear regression.
@model function linear_regression(x, y)
    # Set variance prior.
    σ² ~ truncated(Normal(0, 100); lower=0)

    # Set intercept prior.
    intercept ~ Normal(0, sqrt(3))

    # Set the priors on our coefficients.
    nfeatures = size(x, 2)
    coefficients ~ MvNormal(Zeros(nfeatures), 10.0 * I)

    # Calculate all the mu terms.
    mu = intercept .+ x * coefficients
    return y ~ MvNormal(mu, σ² * I)
end


model = linear_regression(train, train_target)
chain = sample(model, NUTS(), 3_000)


plot(chain)


# Import the GLM package.
using GLM

# Perform multiple regression OLS.
train_with_intercept = hcat(ones(size(train, 1)), train)
ols = lm(train_with_intercept, train_target)

# Compute predictions on the training data set and unstandardize them.
train_prediction_ols = GLM.predict(ols)
StatsBase.reconstruct!(dt_targets, train_prediction_ols)

# Compute predictions on the test data set and unstandardize them.
test_with_intercept = hcat(ones(size(test, 1)), test)
test_prediction_ols = GLM.predict(ols, test_with_intercept)
StatsBase.reconstruct!(dt_targets, test_prediction_ols);


# Make a prediction given an input vector.
function prediction(chain, x)
    p = get_params(chain[200:end, :, :])
    targets = p.intercept' .+ x * reduce(hcat, p.coefficients)'
    return vec(mean(targets; dims=2))
end


# Calculate the predictions for the training and testing sets and unstandardize them.
train_prediction_bayes = prediction(chain, train)
StatsBase.reconstruct!(dt_targets, train_prediction_bayes)
test_prediction_bayes = prediction(chain, test)
StatsBase.reconstruct!(dt_targets, test_prediction_bayes)

# Show the predictions on the test data set.
DataFrame(; MPG=testset[!, target], Bayes=test_prediction_bayes, OLS=test_prediction_ols)


println(
    "Training set:",
    "\n\tBayes loss: ",
    msd(train_prediction_bayes, trainset[!, target]),
    "\n\tOLS loss: ",
    msd(train_prediction_ols, trainset[!, target]),
)

println(
    "Test set:",
    "\n\tBayes loss: ",
    msd(test_prediction_bayes, testset[!, target]),
    "\n\tOLS loss: ",
    msd(test_prediction_ols, testset[!, target]),
)


let
    bayes_train_loss = msd(train_prediction_bayes, trainset[!, target])
    bayes_test_loss = msd(test_prediction_bayes, testset[!, target])
    ols_train_loss = msd(train_prediction_ols, trainset[!, target])
    @assert bayes_train_loss < 5.5
    @assert bayes_test_loss < 15.5
    @assert isapprox(bayes_train_loss, ols_train_loss; atol=1)
end


if isdefined(Main, :TuringTutorials)
    Main.TuringTutorials.tutorial_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])
end



