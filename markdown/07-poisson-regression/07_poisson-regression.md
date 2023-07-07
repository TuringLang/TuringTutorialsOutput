---
redirect_from: "tutorials/7-poissonregression/"
title: "Bayesian Poisson Regression"
permalink: "/tutorials/:name/"
---


This notebook is ported from the [example notebook](https://docs.pymc.io/notebooks/GLM-poisson-regression.html) of PyMC3 on Poisson Regression.

[Poisson Regression](https://en.wikipedia.org/wiki/Poisson_regression) is a technique commonly used to model count data.
Some of the applications include predicting the number of people defaulting on their loans or the number of cars running on a highway on a given day.
This example describes a method to implement the Bayesian version of this technique using Turing.

We will generate the dataset that we will be working on which describes the relationship between number of times a person sneezes during the day with his alcohol consumption and medicinal intake.

We start by importing the required libraries.

```julia
#Import Turing, Distributions and DataFrames
using Turing, Distributions, DataFrames, Distributed

# Import MCMCChain, Plots, and StatsPlots for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots

# Set a seed for reproducibility.
using Random
Random.seed!(12);
```




# Generating data

We start off by creating a toy dataset. We take the case of a person who takes medicine to prevent excessive sneezing. Alcohol consumption increases the rate of sneezing for that person. Thus, the two factors affecting the number of sneezes in a given day are alcohol consumption and whether the person has taken his medicine. Both these variable are taken as boolean valued while the number of sneezes will be a count valued variable. We also take into consideration that the interaction between the two boolean variables will affect the number of sneezes

5 random rows are printed from the generated data to get a gist of the data generated.

```julia
theta_noalcohol_meds = 1    # no alcohol, took medicine
theta_alcohol_meds = 3      # alcohol, took medicine
theta_noalcohol_nomeds = 6  # no alcohol, no medicine
theta_alcohol_nomeds = 36   # alcohol, no medicine

# no of samples for each of the above cases
q = 100

#Generate data from different Poisson distributions
noalcohol_meds = Poisson(theta_noalcohol_meds)
alcohol_meds = Poisson(theta_alcohol_meds)
noalcohol_nomeds = Poisson(theta_noalcohol_nomeds)
alcohol_nomeds = Poisson(theta_alcohol_nomeds)

nsneeze_data = vcat(
    rand(noalcohol_meds, q),
    rand(alcohol_meds, q),
    rand(noalcohol_nomeds, q),
    rand(alcohol_nomeds, q),
)
alcohol_data = vcat(zeros(q), ones(q), zeros(q), ones(q))
meds_data = vcat(zeros(q), zeros(q), ones(q), ones(q))

df = DataFrame(;
    nsneeze=nsneeze_data,
    alcohol_taken=alcohol_data,
    nomeds_taken=meds_data,
    product_alcohol_meds=meds_data .* alcohol_data,
)
df[sample(1:nrow(df), 5; replace=false), :]
```

```
5×4 DataFrame
 Row │ nsneeze  alcohol_taken  nomeds_taken  product_alcohol_meds
     │ Int64    Float64        Float64       Float64
─────┼────────────────────────────────────────────────────────────
   1 │       3            1.0           0.0                   0.0
   2 │       3            0.0           0.0                   0.0
   3 │       7            0.0           1.0                   0.0
   4 │       3            1.0           0.0                   0.0
   5 │       2            0.0           0.0                   0.0
```





# Visualisation of the dataset

We plot the distribution of the number of sneezes for the 4 different cases taken above. As expected, the person sneezes the most when he has taken alcohol and not taken his medicine. He sneezes the least when he doesn't consume alcohol and takes his medicine.

```julia
#Data Plotting

p1 = Plots.histogram(
    df[(df[:, :alcohol_taken] .== 0) .& (df[:, :nomeds_taken] .== 0), 1];
    title="no_alcohol+meds",
)
p2 = Plots.histogram(
    (df[(df[:, :alcohol_taken] .== 1) .& (df[:, :nomeds_taken] .== 0), 1]);
    title="alcohol+meds",
)
p3 = Plots.histogram(
    (df[(df[:, :alcohol_taken] .== 0) .& (df[:, :nomeds_taken] .== 1), 1]);
    title="no_alcohol+no_meds",
)
p4 = Plots.histogram(
    (df[(df[:, :alcohol_taken] .== 1) .& (df[:, :nomeds_taken] .== 1), 1]);
    title="alcohol+no_meds",
)
plot(p1, p2, p3, p4; layout=(2, 2), legend=false)
```

![](figures/07_poisson-regression_3_1.png)



We must convert our `DataFrame` data into the `Matrix` form as the manipulations that we are about are designed to work with `Matrix` data. We also separate the features from the labels which will be later used by the Turing sampler to generate samples from the posterior.

```julia
# Convert the DataFrame object to matrices.
data = Matrix(df[:, [:alcohol_taken, :nomeds_taken, :product_alcohol_meds]])
data_labels = df[:, :nsneeze]
data
```

```
400×3 Matrix{Float64}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 ⋮         
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
```





We must recenter our data about 0 to help the Turing sampler in initialising the parameter estimates. So, normalising the data in each column by subtracting the mean and dividing by the standard deviation:

```julia
# # Rescale our matrices.
data = (data .- mean(data; dims=1)) ./ std(data; dims=1)
```

```
400×3 Matrix{Float64}:
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
 -0.998749  -0.998749  -0.576628
  ⋮                    
  0.998749   0.998749   1.72988
  0.998749   0.998749   1.72988
  0.998749   0.998749   1.72988
  0.998749   0.998749   1.72988
  0.998749   0.998749   1.72988
  0.998749   0.998749   1.72988
  0.998749   0.998749   1.72988
  0.998749   0.998749   1.72988
  0.998749   0.998749   1.72988
```





# Declaring the Model: Poisson Regression

Our model, `poisson_regression` takes four arguments:

  - `x` is our set of independent variables;
  - `y` is the element we want to predict;
  - `n` is the number of observations we have; and
  - `σ²` is the standard deviation we want to assume for our priors.

Within the model, we create four coefficients (`b0`, `b1`, `b2`, and `b3`) and assign a prior of normally distributed with means of zero and standard deviations of `σ²`. We want to find values of these four coefficients to predict any given `y`.

Intuitively, we can think of the coefficients as:

  - `b1` is the coefficient which represents the effect of taking alcohol on the number of sneezes;
  - `b2` is the coefficient which represents the effect of taking in no medicines on the number of sneezes;
  - `b3` is the coefficient which represents the effect of interaction between taking alcohol and no medicine on the number of sneezes;

The `for` block creates a variable `theta` which is the weighted combination of the input features. We have defined the priors on these weights above. We then observe the likelihood of calculating `theta` given the actual label, `y[i]`.

```julia
# Bayesian poisson regression (LR)
@model function poisson_regression(x, y, n, σ²)
    b0 ~ Normal(0, σ²)
    b1 ~ Normal(0, σ²)
    b2 ~ Normal(0, σ²)
    b3 ~ Normal(0, σ²)
    for i in 1:n
        theta = b0 + b1 * x[i, 1] + b2 * x[i, 2] + b3 * x[i, 3]
        y[i] ~ Poisson(exp(theta))
    end
end;
```




# Sampling from the posterior

We use the `NUTS` sampler to sample values from the posterior. We run multiple chains using the `MCMCThreads()` function to nullify the effect of a problematic chain. We then use the Gelman, Rubin, and Brooks Diagnostic to check the convergence of these multiple chains.

```julia
# Retrieve the number of observations.
n, _ = size(data)

# Sample using NUTS.

num_chains = 4
m = poisson_regression(data, data_labels, n, 10)
chain = sample(m, NUTS(), MCMCThreads(), 2_500, num_chains; discard_adapt=false)
```

```
Chains MCMC chain (2500×16×4 Array{Float64, 3}):

Iterations        = 1:1:2500
Number of chains  = 4
Samples per chain = 2500
Wall duration     = 9.82 seconds
Compute duration  = 9.4 seconds
parameters        = b0, b1, b2, b3
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, 
tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat 
  e ⋯
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64 
    ⋯

          b0    1.6309    0.1517     0.0015    0.0089   233.1287    1.0127 
    ⋯
          b1    0.5517    0.0931     0.0009    0.0052   247.0095    1.0164 
    ⋯
          b2    0.8864    0.1101     0.0011    0.0075   173.2922    1.0217 
    ⋯
          b3    0.2950    0.1039     0.0010    0.0050   398.9148    1.0101 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

          b0    1.5763    1.6223    1.6427    1.6632    1.6999
          b1    0.4359    0.5114    0.5510    0.5899    0.6759
          b2    0.7743    0.8440    0.8801    0.9176    0.9970
          b3    0.1770    0.2564    0.2923    0.3290    0.4021
```





# Viewing the Diagnostics

We use the Gelman, Rubin, and Brooks Diagnostic to check whether our chains have converged. Note that we require multiple chains to use this diagnostic which analyses the difference between these multiple chains.

We expect the chains to have converged. This is because we have taken sufficient number of iterations (1500) for the NUTS sampler. However, in case the test fails, then we will have to take a larger number of iterations, resulting in longer computation time.

```julia
gelmandiag(chain)
```

```
Gelman, Rubin, and Brooks diagnostic
  parameters      psrf    psrfci
      Symbol   Float64   Float64

          b0    1.1788    1.2325
          b1    1.0337    1.0556
          b2    1.1912    1.3020
          b3    1.0793    1.0997
```





From the above diagnostic, we can conclude that the chains have converged because the PSRF values of the coefficients are close to 1.

So, we have obtained the posterior distributions of the parameters. We transform the coefficients and recover theta values by taking the exponent of the meaned values of the coefficients `b0`, `b1`, `b2` and `b3`. We take the exponent of the means to get a better comparison of the relative values of the coefficients. We then compare this with the intuitive meaning that was described earlier.

```julia
# Taking the first chain
c1 = chain[:, :, 1]

# Calculating the exponentiated means
b0_exp = exp(mean(c1[:b0]))
b1_exp = exp(mean(c1[:b1]))
b2_exp = exp(mean(c1[:b2]))
b3_exp = exp(mean(c1[:b3]))

print("The exponent of the meaned values of the weights (or coefficients are): \n")
println("b0: ", b0_exp)
println("b1: ", b1_exp)
println("b2: ", b2_exp)
println("b3: ", b3_exp)
print("The posterior distributions obtained after sampling can be visualised as :\n")
```

```
The exponent of the meaned values of the weights (or coefficients are): 
b0: 5.14930308858998
b1: 1.733569117191754
b2: 2.4146446340975065
b3: 1.3430785554075173
The posterior distributions obtained after sampling can be visualised as :
```





Visualising the posterior by plotting it:

```julia
plot(chain)
```

![](figures/07_poisson-regression_10_1.png)



# Interpreting the Obtained Mean Values

The exponentiated mean of the coefficient `b1` is roughly half of that of `b2`. This makes sense because in the data that we generated, the number of sneezes was more sensitive to the medicinal intake as compared to the alcohol consumption. We also get a weaker dependence on the interaction between the alcohol consumption and the medicinal intake as can be seen from the value of `b3`.

# Removing the Warmup Samples

As can be seen from the plots above, the parameters converge to their final distributions after a few iterations.
The initial values during the warmup phase increase the standard deviations of the parameters and are not required after we get the desired distributions.
Thus, we remove these warmup values and once again view the diagnostics.
To remove these warmup values, we take all values except the first 200.
This is because we set the second parameter of the NUTS sampler (which is the number of adaptations) to be equal to 200.

```julia
chains_new = chain[201:end, :, :]
```

```
Chains MCMC chain (2300×16×4 Array{Float64, 3}):

Iterations        = 201:1:2500
Number of chains  = 4
Samples per chain = 2300
Wall duration     = 9.82 seconds
Compute duration  = 9.4 seconds
parameters        = b0, b1, b2, b3
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, 
tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat
    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64
    ⋯

          b0    1.6425    0.0298     0.0003    0.0006   2697.9057    1.0035
    ⋯
          b1    0.5524    0.0572     0.0006    0.0012   2238.1249    1.0027
    ⋯
          b2    0.8820    0.0531     0.0006    0.0011   2125.8687    1.0028
    ⋯
          b3    0.2905    0.0530     0.0006    0.0011   2255.0722    1.0022
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

          b0    1.5820    1.6229    1.6430    1.6630    1.6988
          b1    0.4446    0.5127    0.5522    0.5903    0.6690
          b2    0.7812    0.8448    0.8809    0.9177    0.9903
          b3    0.1828    0.2559    0.2910    0.3277    0.3919
```



```julia
plot(chains_new)
```

![](figures/07_poisson-regression_12_1.png)



As can be seen from the numeric values and the plots above, the standard deviation values have decreased and all the plotted values are from the estimated posteriors. The exponentiated mean values, with the warmup samples removed, have not changed by much and they are still in accordance with their intuitive meanings as described earlier.


if isdefined(Main, :TuringTutorials)
    Main.TuringTutorials.tutorial_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])
end
