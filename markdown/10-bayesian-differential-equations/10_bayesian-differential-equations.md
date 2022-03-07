---
redirect_from: "tutorials/10-bayesiandiffeq/"
title: "Bayesian Estimation of Differential Equations"
permalink: "/:collection/:name/"
---


Most of the scientific community deals with the basic problem of trying to mathematically model the reality around them and this often involves dynamical systems. The general trend to model these complex dynamical systems is through the use of differential equations. Differential equation models often have non-measurable parameters. The popular “forward-problem” of simulation consists of solving the differential equations for a given set of parameters, the “inverse problem” to simulation, known as parameter estimation, is the process of utilizing data to determine these model parameters. Bayesian inference provides a robust approach to parameter estimation with quantified uncertainty.

```julia
using Turing
using DifferentialEquations

# Load StatsPlots for visualizations and diagnostics.
using StatsPlots

# Set a seed for reproducibility.
using Random
Random.seed!(14);
```




## The Lotka-Volterra Model

The Lotka–Volterra equations, also known as the predator–prey equations, are a pair of first-order nonlinear differential equations, frequently used to describe the dynamics of biological systems in which two species interact, one as a predator and the other as prey. The populations change through time according to the pair of equations:

$$\frac{dx}{dt} = (\alpha - \beta y)x$$

$$\frac{dy}{dt} = (\delta x - \gamma)y$$

```julia
function lotka_volterra(du, u, p, t)
    x, y = u
    α, β, γ, δ = p
    du[1] = (α - β * y)x # dx =
    return du[2] = (δ * x - γ)y # dy =
end
p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0, 1.0]
prob1 = ODEProblem(lotka_volterra, u0, (0.0, 10.0), p)
sol = solve(prob1, Tsit5())
plot(sol)
```

![](figures/10_bayesian-differential-equations_2_1.png)



We'll generate the data to use for the parameter estimation from simulation.
With the `saveat` [argument](https://docs.sciml.ai/latest/basics/common_solver_opts/) we specify that the solution is stored only at `0.1` time units. To make the data look more realistic, we add random noise using the function `randn`.

```julia
sol1 = solve(prob1, Tsit5(); saveat=0.1)
odedata = Array(sol1) + 0.8 * randn(size(Array(sol1)))
plot(sol1; alpha=0.3, legend=false);
scatter!(sol1.t, odedata');
```




## Direct Handling of Bayesian Estimation with Turing

Previously, functions in Turing and DifferentialEquations were not inter-composable, so Bayesian inference of differential equations needed to be handled by another package called [DiffEqBayes.jl](https://github.com/SciML/DiffEqBayes.jl) (note that DiffEqBayes works also with CmdStan.jl, Turing.jl, DynamicHMC.jl and ApproxBayes.jl - see the [DiffEqBayes docs](https://docs.sciml.ai/latest/analysis/parameter_estimation/#Bayesian-Methods-1) for more info).

From now on however, Turing and DifferentialEquations are completely composable and we can write of the differential equation inside a Turing `@model` and it will just work. Therefore, we can rewrite the Lotka Volterra parameter estimation problem with a Turing `@model` interface as below:

```julia
Turing.setadbackend(:forwarddiff)

@model function fitlv(data, prob1)
    σ ~ InverseGamma(2, 3) # ~ is the tilde character
    α ~ truncated(Normal(1.5, 0.5), 0.5, 2.5)
    β ~ truncated(Normal(1.2, 0.5), 0, 2)
    γ ~ truncated(Normal(3.0, 0.5), 1, 4)
    δ ~ truncated(Normal(1.0, 0.5), 0, 2)

    p = [α, β, γ, δ]
    prob = remake(prob1; p=p)
    predicted = solve(prob, Tsit5(); saveat=0.1)

    for i in 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ)
    end
end

model = fitlv(odedata, prob1)

# This next command runs 3 independent chains without using multithreading.
chain = sample(model, NUTS(0.65), MCMCSerial(), 1000, 3; progress=false)
```

```
Chains MCMC chain (1000×17×3 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 3
Samples per chain = 1000
Wall duration     = 21.0 seconds
Compute duration  = 20.6 seconds
parameters        = σ, α, β, γ, δ
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, 
tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat
    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64
    ⋯

           σ    0.8120    0.0410     0.0007    0.0011   1734.7161    1.0007
    ⋯
           α    1.5541    0.0538     0.0010    0.0021    793.4515    1.0011
    ⋯
           β    1.0904    0.0533     0.0010    0.0018    915.4801    1.0012
    ⋯
           γ    2.8852    0.1427     0.0026    0.0053    815.7527    1.0009
    ⋯
           δ    0.9403    0.0506     0.0009    0.0019    806.1465    1.0010
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           σ    0.7342    0.7853    0.8099    0.8379    0.8958
           α    1.4551    1.5169    1.5509    1.5888    1.6645
           β    0.9930    1.0533    1.0882    1.1248    1.2030
           γ    2.6111    2.7871    2.8858    2.9821    3.1675
           δ    0.8422    0.9053    0.9404    0.9736    1.0395
```





The estimated parameters are close to the desired parameter values. We can also check that the chains have converged in the plot.

```julia
plot(chain)
```

![](figures/10_bayesian-differential-equations_5_1.png)



### Data retrodiction

In Bayesian analysis it is often useful to retrodict the data, i.e. generate simulated data using samples from the posterior distribution, and compare to the original data (see for instance section 3.3.2 - model checking of McElreath's book "Statistical Rethinking"). Here, we solve again the ODE using the output in `chain`, for 300 randomly picked posterior samples. We plot this ensemble of solutions to check if the solution resembles the data.

```julia
pl = scatter(sol1.t, odedata');
```


```julia
chain_array = Array(chain)
for k in 1:300
    resol = solve(remake(prob1; p=chain_array[rand(1:1500), 1:4]), Tsit5(); saveat=0.1)
    plot!(resol; alpha=0.1, color="#BBBBBB", legend=false)
end
# display(pl)
plot!(sol1; w=1, legend=false)
```

![](figures/10_bayesian-differential-equations_7_1.png)



In the plot above, the 300 retrodicted time courses from the posterior are plotted in gray, and the original data are the blue and red dots, and the solution that was used to generate the data are the green and purple lines. We can see that, even though we added quite a bit of noise to the data (see dot plot above), the posterior distribution reproduces quite accurately the "true" ODE solution.

## Lotka Volterra with missing predator data

Thanks to the known structure of the problem, encoded by the Lokta-Volterra ODEs, one can also fit a model with incomplete data - even without any data for one of the two variables. For instance, let's suppose you have observations for the prey only, but none for the predator. We test this case by fitting the model only to the $$y$$ variable of the system, without providing any data for $$x$$:

```julia
@model function fitlv2(data, prob1) # data should be a Vector
    σ ~ InverseGamma(2, 3) # ~ is the tilde character
    α ~ truncated(Normal(1.5, 0.5), 0.5, 2.5)
    β ~ truncated(Normal(1.2, 0.5), 0, 2)
    γ ~ truncated(Normal(3.0, 0.5), 1, 4)
    δ ~ truncated(Normal(1.0, 0.5), 0, 2)

    p = [α, β, γ, δ]
    prob = remake(prob1; p=p)
    predicted = solve(prob, Tsit5(); saveat=0.1)

    for i in 1:length(predicted)
        data[i] ~ Normal(predicted[i][2], σ) # predicted[i][2] is the data for y - a scalar, so we use Normal instead of MvNormal
    end
end

model2 = fitlv2(odedata[2, :], prob1)
```

```
DynamicPPL.Model{typeof(Main.##WeaveSandBox#4810.fitlv2), (:data, :prob1), 
(), (), Tuple{Vector{Float64}, SciMLBase.ODEProblem{Vector{Float64}, Tuple{
Float64, Float64}, true, Vector{Float64}, SciMLBase.ODEFunction{true, typeo
f(Main.##WeaveSandBox#4810.lotka_volterra), LinearAlgebra.UniformScaling{Bo
ol}, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing
, Nothing, Nothing, Nothing, Nothing, typeof(SciMLBase.DEFAULT_OBSERVED), N
othing}, Base.Iterators.Pairs{Union{}, Union{}, Tuple{}, NamedTuple{(), Tup
le{}}}, SciMLBase.StandardODEProblem}}, Tuple{}, DynamicPPL.DefaultContext}
(:fitlv2, Main.##WeaveSandBox#4810.fitlv2, (data = [2.200730590544725, 0.85
84002186440604, 0.31308038923384407, 0.8065538543184619, -0.347195243796585
1, 0.2827563462601048, 0.46337329091344126, 0.9388139946097066, -0.02963888
8419957654, -0.10766570796447789  …  4.4844669073068015, 2.2766378547092803
, 3.034635398109275, 1.6534146147281994, 2.3126757947633196, 3.430419239300
9023, 1.481768351221499, 1.7989355388635422, 1.3438819631213252, 0.25843622
408034894], prob1 = SciMLBase.ODEProblem{Vector{Float64}, Tuple{Float64, Fl
oat64}, true, Vector{Float64}, SciMLBase.ODEFunction{true, typeof(Main.##We
aveSandBox#4810.lotka_volterra), LinearAlgebra.UniformScaling{Bool}, Nothin
g, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, 
Nothing, Nothing, Nothing, typeof(SciMLBase.DEFAULT_OBSERVED), Nothing}, Ba
se.Iterators.Pairs{Union{}, Union{}, Tuple{}, NamedTuple{(), Tuple{}}}, Sci
MLBase.StandardODEProblem}(SciMLBase.ODEFunction{true, typeof(Main.##WeaveS
andBox#4810.lotka_volterra), LinearAlgebra.UniformScaling{Bool}, Nothing, N
othing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Noth
ing, Nothing, Nothing, typeof(SciMLBase.DEFAULT_OBSERVED), Nothing}(Main.##
WeaveSandBox#4810.lotka_volterra, LinearAlgebra.UniformScaling{Bool}(true),
 nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, no
thing, nothing, nothing, nothing, SciMLBase.DEFAULT_OBSERVED, nothing), [1.
0, 1.0], (0.0, 10.0), [1.5, 1.0, 3.0, 1.0], Base.Iterators.Pairs{Union{}, U
nion{}, Tuple{}, NamedTuple{(), Tuple{}}}(), SciMLBase.StandardODEProblem()
)), NamedTuple(), DynamicPPL.DefaultContext())
```





Here we use the multithreading functionality [available](https://turing.ml/dev/docs/using-turing/guide#multithreaded-sampling) in Turing.jl to sample 3 independent chains

```julia
Threads.nthreads()
```

```
1
```



```julia
# This next command runs 3 independent chains with multithreading.
chain2 = sample(model2, NUTS(0.45), MCMCThreads(), 5000, 3; progress=false)
```

```
Chains MCMC chain (5000×17×3 Array{Float64, 3}):

Iterations        = 1001:1:6000
Number of chains  = 3
Samples per chain = 5000
Wall duration     = 50.75 seconds
Compute duration  = 50.3 seconds
parameters        = σ, α, β, γ, δ
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, 
tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat 
  e ⋯
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64 
    ⋯

           σ    0.8215    0.0580     0.0005    0.0032   219.1185    1.0108 
    ⋯
           α    1.5579    0.1988     0.0016    0.0109   187.2055    1.0051 
    ⋯
           β    1.1209    0.1573     0.0013    0.0083   217.0800    1.0046 
    ⋯
           γ    2.9855    0.3210     0.0026    0.0178   188.2175    1.0031 
    ⋯
           δ    0.9573    0.2630     0.0021    0.0144   199.7913    1.0034 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           σ    0.7155    0.7815    0.8180    0.8579    0.9444
           α    1.2459    1.3959    1.5409    1.6922    1.9914
           β    0.8669    0.9973    1.1095    1.2271    1.4616
           γ    2.4119    2.7473    2.9707    3.2380    3.5832
           δ    0.4930    0.7562    0.9376    1.1575    1.4797
```



```julia
pl = scatter(sol1.t, odedata');
chain_array2 = Array(chain2)
for k in 1:300
    resol = solve(remake(prob1; p=chain_array2[rand(1:12000), 1:4]), Tsit5(); saveat=0.1)
    # Note that due to a bug in AxisArray, the variables from the chain will be returned always in
    # the order it is stored in the array, not by the specified order in the call - :α, :β, :γ, :δ
    plot!(resol; alpha=0.1, color="#BBBBBB", legend=false)
end
#display(pl)
plot!(sol1; w=1, legend=false)
```

![](figures/10_bayesian-differential-equations_11_1.png)



Note that here, the data values of $$x$$ (blue dots) were not given to the model! Yet, the model could predict the values of $$x$$ relatively accurately, albeit with a wider distribution of solutions, reflecting the greater uncertainty in the prediction of the $$x$$ values.

## Inference of Delay Differential Equations

Here we show an example of inference with another type of differential equation: a Delay Differential Equation (DDE). A DDE is an DE system where derivatives are function of values at an earlier point in time. This is useful to model a delayed effect, like incubation time of a virus for instance.

For this, we will define a [`DDEProblem`](https://diffeq.sciml.ai/stable/tutorials/dde_example/), from the package DifferentialEquations.jl.

Here is a delayed version of the lokta voltera system:

$$\frac{dx}{dt} = \alpha x(t-\tau) - \beta y(t) x(t)$$

$$\frac{dy}{dt} = - \gamma y(t) + \delta x(t) y(t) $$

Where $$x(t-\tau)$$ is the variable $$x$$ at an earlier time point. We specify the delayed variable with a function `h(p, t)`, as described in the [DDE example](https://diffeq.sciml.ai/stable/tutorials/dde_example/).

```julia
function delay_lotka_volterra(du, u, h, p, t)
    x, y = u
    α, β, γ, δ = p
    du[1] = α * h(p, t - 1; idxs=1) - β * x * y
    du[2] = -γ * y + δ * x * y
    return nothing
end

p = (1.5, 1.0, 3.0, 1.0)
u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
h(p, t; idxs::Int) = 1.0
prob1 = DDEProblem(delay_lotka_volterra, u0, h, tspan, p)
```

```
DDEProblem with uType Vector{Float64} and tType Float64. In-place: true
timespan: (0.0, 10.0)
u0: 2-element Vector{Float64}:
 1.0
 1.0
```



```julia
sol = solve(prob1; saveat=0.1)
ddedata = Array(sol)
ddedata = ddedata + 0.5 * randn(size(ddedata))
```

```
2×101 Matrix{Float64}:
 1.21203  2.3054    1.09458  0.339482  …  2.69834  2.71952  1.99324  3.0851
3
 1.72486  0.798932  1.17561  0.335901     1.78836  2.76679  2.02838  1.1793
4
```





Plot the data:

```julia
scatter(sol.t, ddedata');
plot!(sol);
```




Now we define and run the Turing model.

```julia
Turing.setadbackend(:forwarddiff)
@model function fitlv(data, prob1)
    σ ~ InverseGamma(2, 3)
    α ~ Truncated(Normal(1.5, 0.5), 0.5, 2.5)
    β ~ Truncated(Normal(1.2, 0.5), 0, 2)
    γ ~ Truncated(Normal(3.0, 0.5), 1, 4)
    δ ~ Truncated(Normal(1.0, 0.5), 0, 2)

    p = [α, β, γ, δ]

    #prob = DDEProblem(delay_lotka_volterra,u0,_h,tspan,p)
    prob = remake(prob1; p=p)
    predicted = solve(prob; saveat=0.1)
    for i in 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ)
    end
end;
model = fitlv(ddedata, prob1)
```

```
DynamicPPL.Model{typeof(Main.##WeaveSandBox#4810.fitlv), (:data, :prob1), (
), (), Tuple{Matrix{Float64}, SciMLBase.DDEProblem{Vector{Float64}, Tuple{F
loat64, Float64}, Tuple{}, Tuple{}, true, NTuple{4, Float64}, SciMLBase.DDE
Function{true, typeof(Main.##WeaveSandBox#4810.delay_lotka_volterra), Linea
rAlgebra.UniformScaling{Bool}, Nothing, Nothing, Nothing, Nothing, Nothing,
 Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, typeof(SciMLBase.DEF
AULT_OBSERVED), Nothing}, typeof(Main.##WeaveSandBox#4810.h), Base.Iterator
s.Pairs{Union{}, Union{}, Tuple{}, NamedTuple{(), Tuple{}}}, SciMLBase.Stan
dardDDEProblem}}, Tuple{}, DynamicPPL.DefaultContext}(:fitlv, Main.##WeaveS
andBox#4810.fitlv, (data = [1.2120271584754332 2.305404804595023 … 1.993238
2992164426 3.085134499747149; 1.724864558721979 0.7989319537137783 … 2.0283
785588708914 1.179341898716979], prob1 = SciMLBase.DDEProblem{Vector{Float6
4}, Tuple{Float64, Float64}, Tuple{}, Tuple{}, true, NTuple{4, Float64}, Sc
iMLBase.DDEFunction{true, typeof(Main.##WeaveSandBox#4810.delay_lotka_volte
rra), LinearAlgebra.UniformScaling{Bool}, Nothing, Nothing, Nothing, Nothin
g, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, typeof(Sc
iMLBase.DEFAULT_OBSERVED), Nothing}, typeof(Main.##WeaveSandBox#4810.h), Ba
se.Iterators.Pairs{Union{}, Union{}, Tuple{}, NamedTuple{(), Tuple{}}}, Sci
MLBase.StandardDDEProblem}(SciMLBase.DDEFunction{true, typeof(Main.##WeaveS
andBox#4810.delay_lotka_volterra), LinearAlgebra.UniformScaling{Bool}, Noth
ing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing
, Nothing, Nothing, typeof(SciMLBase.DEFAULT_OBSERVED), Nothing}(Main.##Wea
veSandBox#4810.delay_lotka_volterra, LinearAlgebra.UniformScaling{Bool}(tru
e), nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
 nothing, nothing, nothing, SciMLBase.DEFAULT_OBSERVED, nothing), [1.0, 1.0
], Main.##WeaveSandBox#4810.h, (0.0, 10.0), (1.5, 1.0, 3.0, 1.0), (), (), B
ase.Iterators.Pairs{Union{}, Union{}, Tuple{}, NamedTuple{(), Tuple{}}}(), 
false, 0, SciMLBase.StandardDDEProblem())), NamedTuple(), DynamicPPL.Defaul
tContext())
```





Then we draw samples using multithreading; this time, we draw 3 independent chains in parallel using `MCMCThreads`.

```julia
chain = sample(model, NUTS(0.65), MCMCThreads(), 300, 3; progress=false)
```

```
Chains MCMC chain (300×17×3 Array{Float64, 3}):

Iterations        = 151:1:450
Number of chains  = 3
Samples per chain = 300
Wall duration     = 42.7 seconds
Compute duration  = 42.34 seconds
parameters        = σ, α, β, γ, δ
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, 
tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat 
  e ⋯
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64 
    ⋯

           σ    0.4922    0.0268     0.0009    0.0013   353.3690    1.0036 
    ⋯
           α    1.4424    0.0573     0.0019    0.0031   280.5332    1.0140 
    ⋯
           β    0.9770    0.0453     0.0015    0.0020   340.6927    1.0086 
    ⋯
           γ    3.1517    0.1428     0.0048    0.0076   338.7052    1.0141 
    ⋯
           δ    1.0638    0.0504     0.0017    0.0028   312.0792    1.0146 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           σ    0.4435    0.4738    0.4916    0.5088    0.5492
           α    1.3356    1.4035    1.4399    1.4814    1.5607
           β    0.8952    0.9457    0.9754    1.0065    1.0671
           γ    2.8862    3.0607    3.1467    3.2350    3.4392
           δ    0.9703    1.0309    1.0636    1.0925    1.1671
```



```julia
plot(chain)
```

![](figures/10_bayesian-differential-equations_17_1.png)



Finally, we select a 100 sets of parameters from the first chain and plot solutions.

```julia
pl = scatter(sol.t, ddedata')
chain_array = Array(chain)
for k in 1:100
    resol = solve(remake(prob1; p=chain_array[rand(1:450), 1:4]), Tsit5(); saveat=0.1)
    # Note that due to a bug in AxisArray, the variables from the chain will be returned always in
    # the order it is stored in the array, not by the specified order in the call - :α, :β, :γ, :δ

    plot!(resol; alpha=0.1, color="#BBBBBB", legend=false)
end
#display(pl)
plot!(sol)
```

![](figures/10_bayesian-differential-equations_18_1.png)



Here again, the dots is the data fed to the model, the continuous colored line is the "true" solution, and the gray lines are solutions from the posterior. The fit is pretty good even though the data was quite noisy to start.

## Scaling to Large Models: Adjoint Sensitivities

DifferentialEquations.jl's efficiency for large stiff models has been shown in multiple [benchmarks](https://github.com/SciML/DiffEqBenchmarks.jl). To learn more about how to optimize solving performance for stiff problems you can take a look at the [docs](https://docs.sciml.ai/latest/tutorials/advanced_ode_example/).

[Sensitivity analysis](https://docs.sciml.ai/latest/analysis/sensitivity/), or automatic differentiation (AD) of the solver, is provided by the DiffEq suite. The model sensitivities are the derivatives of the solution $$u(t)$$ with respect to the parameters. Specifically, the local sensitivity of the solution to a parameter is defined by how much the solution would change by changes in the parameter. Sensitivity analysis provides a cheap way to calculate the gradient of the solution which can be used in parameter estimation and other optimization tasks.

The AD ecosystem in Julia allows you to switch between forward mode, reverse mode, source to source and other choices of AD and have it work with any Julia code. For a user to make use of this within [SciML](https://sciml.ai), [high level interactions in `solve`](https://docs.sciml.ai/latest/analysis/sensitivity/#High-Level-Interface:-sensealg-1) automatically plug into those AD systems to allow for choosing advanced sensitivity analysis (derivative calculation) [methods](https://docs.sciml.ai/latest/analysis/sensitivity/#Sensitivity-Algorithms-1).

More theoretical details on these methods can be found at: https://docs.sciml.ai/latest/extras/sensitivity_math/.

While these sensitivity analysis methods may seem complicated (and they are!), using them is dead simple. Here is a version of the Lotka-Volterra model with adjoints enabled.

All we had to do is switch the AD backend to one of the adjoint-compatible backends (ReverseDiff, Tracker, or Zygote) and boom the system takes over and we're using adjoint methods! Notice that on this model adjoints are slower. This is because adjoints have a higher overhead on small parameter models and we suggest only using these methods for models with around 100 parameters or more. For more details, see https://arxiv.org/abs/1812.01892.

```julia
using Zygote, DiffEqSensitivity
Turing.setadbackend(:zygote)
prob1 = ODEProblem(lotka_volterra, u0, (0.0, 10.0), p)
```

```
ODEProblem with uType Vector{Float64} and tType Float64. In-place: true
timespan: (0.0, 10.0)
u0: 2-element Vector{Float64}:
 1.0
 1.0
```



```julia
@model function fitlv(data, prob)
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5, 0.5), 0.5, 2.5)
    β ~ truncated(Normal(1.2, 0.5), 0, 2)
    γ ~ truncated(Normal(3.0, 0.5), 1, 4)
    δ ~ truncated(Normal(1.0, 0.5), 0, 2)
    p = [α, β, γ, δ]
    prob = remake(prob; p=p)

    predicted = solve(prob; saveat=0.1)
    for i in 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ)
    end
end;
model = fitlv(odedata, prob1)
chain = sample(model, NUTS(0.65), 1000)
```

```
Chains MCMC chain (1000×17×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 959.03 seconds
Compute duration  = 959.03 seconds
parameters        = σ, α, β, γ, δ
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, 
tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat 
  e ⋯
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64 
    ⋯

           σ    0.8138    0.0406     0.0013    0.0015   554.6180    0.9990 
    ⋯
           α    1.5606    0.0518     0.0016    0.0038   159.0399    1.0065 
    ⋯
           β    1.0976    0.0532     0.0017    0.0030   298.2408    1.0084 
    ⋯
           γ    2.8667    0.1364     0.0043    0.0099   159.1570    1.0052 
    ⋯
           δ    0.9342    0.0492     0.0016    0.0036   161.8787    1.0067 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           σ    0.7434    0.7849    0.8123    0.8398    0.8941
           α    1.4686    1.5241    1.5603    1.5961    1.6678
           β    0.9997    1.0617    1.0936    1.1323    1.2045
           γ    2.6050    2.7730    2.8655    2.9589    3.1444
           δ    0.8443    0.8987    0.9312    0.9690    1.0286
```





Now we can exercise control of the sensitivity analysis method that is used by using the `sensealg` keyword argument. Let's choose the `InterpolatingAdjoint` from the available AD [methods](https://docs.sciml.ai/latest/analysis/sensitivity/#Sensitivity-Algorithms-1) and enable a compiled ReverseDiff vector-Jacobian product:

```julia
@model function fitlv(data, prob)
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5, 0.5), 0.5, 2.5)
    β ~ truncated(Normal(1.2, 0.5), 0, 2)
    γ ~ truncated(Normal(3.0, 0.5), 1, 4)
    δ ~ truncated(Normal(1.0, 0.5), 0, 2)
    p = [α, β, γ, δ]
    prob = remake(prob; p=p)
    predicted = solve(
        prob; saveat=0.1, sensealg=InterpolatingAdjoint(; autojacvec=ReverseDiffVJP(true))
    )
    for i in 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ)
    end
end;
model = fitlv(odedata, prob1)
@time chain = sample(model, NUTS(0.65), 1000; progress=false)
```

```
932.534760 seconds (4.22 G allocations: 635.686 GiB, 9.92% gc time, 1.38% c
ompilation time)
Chains MCMC chain (1000×17×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 929.2 seconds
Compute duration  = 929.2 seconds
parameters        = σ, α, β, γ, δ
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, 
tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat 
  e ⋯
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64 
    ⋯

           σ    0.8104    0.0402     0.0013    0.0017   514.9641    0.9993 
    ⋯
           α    1.5564    0.0518     0.0016    0.0034   202.7899    1.0105 
    ⋯
           β    1.0909    0.0516     0.0016    0.0033   213.6105    1.0090 
    ⋯
           γ    2.8787    0.1383     0.0044    0.0086   216.2721    1.0083 
    ⋯
           δ    0.9373    0.0487     0.0015    0.0032   202.1714    1.0111 
    ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           σ    0.7375    0.7844    0.8106    0.8349    0.8884
           α    1.4643    1.5219    1.5523    1.5872    1.6700
           β    1.0001    1.0545    1.0890    1.1233    1.2019
           γ    2.5994    2.7889    2.8846    2.9629    3.1415
           δ    0.8405    0.9069    0.9364    0.9691    1.0315
```





For more examples of adjoint usage on large parameter models, consult the [DiffEqFlux documentation](https://diffeqflux.sciml.ai/dev/).

## Inference of a Stochastic Differential Equation

A Stochastic Differential Equation ([SDE](https://diffeq.sciml.ai/stable/tutorials/sde_example/)) is a differential equation that has a stochastic (noise) term in the expression of the derivatives. Here we fit a Stochastic version of the Lokta-Volterra system.

We use a quasi-likelihood approach in which all trajectories of a solution are compared instead of a reduction such as mean, this increases the robustness of fitting and makes the likelihood more identifiable. We use SOSRI to solve the equation. The NUTS sampler is a bit sensitive to the stochastic optimization since the gradient is then changing with every calculation, so we use NUTS with a target acceptance rate of `0.25`.

```julia
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
function multiplicative_noise!(du, u, p, t)
    x, y = u
    du[1] = p[5] * x
    return du[2] = p[6] * y
end
p = [1.5, 1.0, 3.0, 1.0, 0.1, 0.1]

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, γ, δ = p
    du[1] = dx = α * x - β * x * y
    return du[2] = dy = δ * x * y - γ * y
end

prob_sde = SDEProblem(lotka_volterra!, multiplicative_noise!, u0, tspan, p)

ensembleprob = EnsembleProblem(prob_sde)
@time data = solve(ensembleprob, SOSRI(); saveat=0.1, trajectories=1000)
plot(EnsembleSummary(data))
```

```
6.515669 seconds (6.01 M allocations: 315.441 MiB, 1.77% gc time, 64.22% 
compilation time)
```


![](figures/10_bayesian-differential-equations_22_1.png)

```julia
Turing.setadbackend(:forwarddiff)
@model function fitlv(data, prob)
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.3, 0.5), 0.5, 2.5)
    β ~ truncated(Normal(1.2, 0.25), 0.5, 2)
    γ ~ truncated(Normal(3.2, 0.25), 2.2, 4.0)
    δ ~ truncated(Normal(1.2, 0.25), 0.5, 2.0)
    ϕ1 ~ truncated(Normal(0.12, 0.3), 0.05, 0.25)
    ϕ2 ~ truncated(Normal(0.12, 0.3), 0.05, 0.25)
    p = [α, β, γ, δ, ϕ1, ϕ2]
    prob = remake(prob; p=p)
    predicted = solve(prob, SOSRI(); saveat=0.1)

    if predicted.retcode !== :Success
        Turing.@addlogprob! -Inf
    end

    for i in 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ)
    end
end;
```




We use NUTS sampler with a low acceptance ratio and initial parameters since estimating the parameters of SDE with HMC poses a challenge. Probabilistic nature of the SDE solution makes the likelihood function noisy which poses a challenge for NUTS since the gradient is then changing with every calculation. SGHMC might be better suited to be used here.

```julia
model = fitlv(odedata, prob_sde)
chain = sample(model, NUTS(0.25), 5000; init_params=[1.5, 1.3, 1.2, 2.7, 1.2, 0.12, 0.12])
plot(chain)
```

![](figures/10_bayesian-differential-equations_24_1.png)


## Appendix

These tutorials are a part of the TuringTutorials repository, found at: [https://github.com/TuringLang/TuringTutorials](https://github.com/TuringLang/TuringTutorials).

To locally run this tutorial, do the following commands:

```
using TuringTutorials
TuringTutorials.weave("10-bayesian-differential-equations", "10_bayesian-differential-equations.jmd")
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
  JULIA_CPU_THREADS = 16
  BUILDKITE_PLUGIN_JULIA_CACHE_DIR = /cache/julia-buildkite-plugin
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/7aa0085e-79a4-45f3-a5bd-9743c91cf3da

```

Package Information:

```
      Status `/cache/build/default-amdci4-7/julialang/turingtutorials/tutorials/10-bayesian-differential-equations/Project.toml`
  [41bf760c] DiffEqSensitivity v6.70.0
  [0c46a032] DifferentialEquations v7.1.0
  [f3b207a7] StatsPlots v0.14.33
  [fce5fe82] Turing v0.20.4
  [e88e6eb3] Zygote v0.6.35
  [9a3f8284] Random
```

And the full manifest:

```
      Status `/cache/build/default-amdci4-7/julialang/turingtutorials/tutorials/10-bayesian-differential-equations/Manifest.toml`
  [621f4979] AbstractFFTs v1.1.0
  [80f14c24] AbstractMCMC v3.3.1
  [7a57a42e] AbstractPPL v0.5.1
  [1520ce14] AbstractTrees v0.3.4
  [79e6a3ab] Adapt v3.3.3
  [0bf59076] AdvancedHMC v0.3.3
  [5b7e9947] AdvancedMH v0.6.6
  [576499cb] AdvancedPS v0.3.5
  [b5ca4192] AdvancedVI v0.1.3
  [dce04be8] ArgCheck v2.3.0
  [ec485272] ArnoldiMethod v0.2.0
  [7d9fca2a] Arpack v0.5.3
  [4fba245c] ArrayInterface v3.2.2
  [4c555306] ArrayLayouts v0.7.10
  [13072b0f] AxisAlgorithms v1.0.1
  [39de3d68] AxisArrays v0.4.4
  [aae01518] BandedMatrices v0.16.11
  [198e06fe] BangBang v0.3.36
  [9718e550] Baselet v0.1.1
  [76274a88] Bijectors v0.9.11
  [62783981] BitTwiddlingConvenienceFunctions v0.1.3
  [8e7c35d0] BlockArrays v0.16.11
  [ffab5731] BlockBandedMatrices v0.11.1
  [764a87c0] BoundaryValueDiffEq v2.7.1
  [fa961155] CEnum v0.4.1
  [2a0fbf3d] CPUSummary v0.1.12
  [49dc2e85] Calculus v0.5.1
  [7057c7e9] Cassette v0.3.9
  [082447d4] ChainRules v1.27.0
  [d360d2e6] ChainRulesCore v1.13.0
  [9e997f8a] ChangesOfVariables v0.1.2
  [fb6a15b2] CloseOpenIntervals v0.1.6
  [aaaa29a8] Clustering v0.14.2
  [35d6a980] ColorSchemes v3.17.1
  [3da002f7] ColorTypes v0.11.0
  [5ae59095] Colors v0.12.8
  [861a8166] Combinatorics v1.0.2
  [38540f10] CommonSolve v0.2.0
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v3.41.0
  [b152e2b5] CompositeTypes v0.1.2
  [a33af91c] CompositionsBase v0.1.1
  [88cd18e8] ConsoleProgressMonitor v0.1.2
  [187b0558] ConstructionBase v1.3.0
  [d38c429a] Contour v0.5.7
  [a8cc5b0e] Crayons v4.1.1
  [754358af] DEDataArrays v0.2.2
  [9a962f9c] DataAPI v1.9.0
  [864edb3b] DataStructures v0.18.11
  [e2d170a0] DataValueInterfaces v1.0.0
  [e7dc6d0d] DataValues v0.4.13
  [244e2a9f] DefineSingletons v0.1.2
  [bcd4f6db] DelayDiffEq v5.35.0
  [b429d917] DensityInterface v0.4.0
  [2b5f629d] DiffEqBase v6.82.0
  [459566f4] DiffEqCallbacks v2.22.0
  [c894b116] DiffEqJump v8.3.0
  [77a26b50] DiffEqNoiseProcess v5.9.0
  [9fdde737] DiffEqOperators v4.41.0
  [41bf760c] DiffEqSensitivity v6.70.0
  [163ba53b] DiffResults v1.0.3
  [b552c78f] DiffRules v1.10.0
  [0c46a032] DifferentialEquations v7.1.0
  [b4f34e82] Distances v0.10.7
  [31c24e10] Distributions v0.25.49
  [ced4e74d] DistributionsAD v0.6.38
  [ffbed154] DocStringExtensions v0.8.6
  [5b8099bc] DomainSets v0.5.9
  [fa6b7ba4] DualNumbers v0.6.6
  [366bfd00] DynamicPPL v0.17.8
  [da5c29d0] EllipsisNotation v1.3.0
  [cad2338a] EllipticalSliceSampling v0.4.7
  [7da242da] Enzyme v0.8.5
  [d4d017d3] ExponentialUtilities v1.13.0
  [e2ba6199] ExprTools v0.1.8
  [c87230d0] FFMPEG v0.4.1
  [7a1cc6ca] FFTW v1.4.6
  [7034ab61] FastBroadcast v0.1.14
  [9aa1b823] FastClosures v0.3.2
  [1a297f60] FillArrays v0.12.8
  [6a86dc24] FiniteDiff v2.11.0
  [53c48c17] FixedPointNumbers v0.8.4
  [59287772] Formatting v0.4.2
  [f6369f11] ForwardDiff v0.10.25
  [069b7b12] FunctionWrappers v1.1.2
  [d9f16b24] Functors v0.2.8
  [61eb1bfa] GPUCompiler v0.13.14
  [28b8d3ca] GR v0.64.0
  [5c1252a2] GeometryBasics v0.4.2
  [af5da776] GlobalSensitivity v1.3.1
  [86223c79] Graphs v1.6.0
  [42e2da0e] Grisu v1.0.2
  [cd3eb016] HTTP v0.9.17
  [3e5b6fbb] HostCPUFeatures v0.1.7
  [0e44f5e4] Hwloc v2.0.0
  [34004b35] HypergeometricFunctions v0.3.8
  [7869d1d1] IRTools v0.4.5
  [615f187c] IfElse v0.1.1
  [d25df0c9] Inflate v0.1.2
  [83e8ac13] IniFile v0.5.1
  [22cec73e] InitialValues v0.3.1
  [505f98c9] InplaceOps v0.3.0
  [a98d9a8b] Interpolations v0.13.5
  [8197267c] IntervalSets v0.5.3
  [3587e190] InverseFunctions v0.1.2
  [41ab1584] InvertedIndices v1.1.0
  [92d709cd] IrrationalConstants v0.1.1
  [c8e1da08] IterTools v1.4.0
  [42fd0dbc] IterativeSolvers v0.9.2
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.4.1
  [682c06a0] JSON v0.21.3
  [ef3ab10e] KLU v0.3.0
  [5ab0869b] KernelDensity v0.6.3
  [ba0b0d4f] Krylov v0.7.13
  [0b1a1467] KrylovKit v0.5.3
  [929cbde3] LLVM v4.9.0
  [8ac3fa9e] LRUCache v1.3.0
  [b964fa9f] LaTeXStrings v1.3.0
  [2ee39098] LabelledArrays v1.8.0
  [23fbe1c1] Latexify v0.15.12
  [a5e1c1ea] LatinHypercubeSampling v1.8.0
  [73f95e8e] LatticeRules v0.0.1
  [10f19ff3] LayoutPointers v0.1.6
  [5078a376] LazyArrays v0.22.5
  [d7e5e226] LazyBandedMatrices v0.7.6
  [1d6d02ad] LeftChildRightSiblingTrees v0.1.3
  [2d8b4e74] LevyArea v1.0.0
  [6f1fad26] Libtask v0.6.10
  [d3d80556] LineSearches v7.1.1
  [7ed4a6bd] LinearSolve v1.13.0
  [2ab3a3ac] LogExpFunctions v0.3.6
  [e6f89c97] LoggingExtras v0.4.7
  [bdcacae8] LoopVectorization v0.12.103
  [c7f686f2] MCMCChains v5.0.4
  [be115224] MCMCDiagnosticTools v0.1.3
  [e80e1ace] MLJModelInterface v1.4.1
  [1914dd2f] MacroTools v0.5.9
  [d125e4d3] ManualMemory v0.1.8
  [dbb5928d] MappedArrays v0.4.1
  [a3b82374] MatrixFactorizations v0.8.5
  [739be429] MbedTLS v1.0.3
  [442fdcdd] Measures v0.3.1
  [128add7d] MicroCollections v0.1.2
  [e1d29d7a] Missings v1.0.2
  [46d2c3a1] MuladdMacro v0.2.2
  [6f286f6a] MultivariateStats v0.9.1
  [d41bc354] NLSolversBase v7.8.2
  [2774e3e8] NLsolve v4.5.1
  [872c559c] NNlib v0.8.3
  [77ba4419] NaNMath v0.3.7
  [86f7a689] NamedArrays v0.9.6
  [c020b1a1] NaturalSort v1.0.0
  [b8a86587] NearestNeighbors v0.4.9
  [8913a72c] NonlinearSolve v0.3.15
  [d8793406] ObjectFile v0.3.7
  [510215fc] Observables v0.4.0
  [6fe1bfb0] OffsetArrays v1.10.8
  [429524aa] Optim v1.6.2
  [bac558e1] OrderedCollections v1.4.1
  [1dea7af3] OrdinaryDiffEq v6.7.0
  [90014a1f] PDMats v0.11.6
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.2.2
  [ccf2f8ad] PlotThemes v2.0.1
  [995b91a9] PlotUtils v1.1.3
  [91a5bcdd] Plots v1.26.0
  [e409e4f3] PoissonRandom v0.4.0
  [f517fe37] Polyester v0.6.7
  [1d0040c9] PolyesterWeave v0.1.5
  [85a6dd25] PositiveFactorizations v0.2.4
  [d236fae5] PreallocationTools v0.2.4
  [21216c6a] Preferences v1.2.4
  [08abe8d2] PrettyTables v1.3.1
  [33c8b6b6] ProgressLogging v0.1.4
  [92933f4c] ProgressMeter v1.7.1
  [1fd47b50] QuadGK v2.4.2
  [8a4e6c94] QuasiMonteCarlo v0.2.4
  [74087812] Random123 v1.5.0
  [e6cf234a] RandomNumbers v1.5.3
  [b3c3ace0] RangeArrays v0.3.2
  [c84ed2f1] Ratios v0.4.3
  [c1ae055f] RealDot v0.1.0
  [3cdcf5f2] RecipesBase v1.2.1
  [01d81517] RecipesPipeline v0.5.1
  [731186ca] RecursiveArrayTools v2.25.0
  [f2c3362d] RecursiveFactorization v0.2.9
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v0.1.3
  [ae029012] Requires v1.3.0
  [ae5879a3] ResettableStacks v1.1.1
  [37e2e3b7] ReverseDiff v1.12.0
  [79098fc4] Rmath v0.7.0
  [f2b01f46] Roots v1.3.14
  [7e49a35a] RuntimeGeneratedFunctions v0.5.3
  [3cdde19b] SIMDDualNumbers v0.1.0
  [94e857df] SIMDTypes v0.1.0
  [476501e8] SLEEFPirates v0.6.31
  [0bca4576] SciMLBase v1.28.0
  [30f210dd] ScientificTypesBase v3.0.0
  [6c6a2e73] Scratch v1.1.0
  [91c51154] SentinelArrays v1.3.12
  [efcf1570] Setfield v0.8.2
  [992d4aef] Showoff v1.0.3
  [699a6c99] SimpleTraits v0.9.4
  [ed01d8cd] Sobol v1.5.0
  [a2af1166] SortingAlgorithms v1.0.1
  [47a9eef4] SparseDiffTools v1.20.2
  [276daf66] SpecialFunctions v2.1.4
  [171d559e] SplittablesBase v0.1.14
  [860ef19b] StableRNGs v1.0.0
  [aedffcd0] Static v0.4.1
  [90137ffa] StaticArrays v1.4.1
  [64bff920] StatisticalTraits v3.0.0
  [82ae8749] StatsAPI v1.2.1
  [2913bbd2] StatsBase v0.33.16
  [4c63d2b9] StatsFuns v0.9.16
  [f3b207a7] StatsPlots v0.14.33
  [9672c7b4] SteadyStateDiffEq v1.6.6
  [789caeaf] StochasticDiffEq v6.45.0
  [7792a7ef] StrideArraysCore v0.2.13
  [09ab397b] StructArrays v0.6.5
  [53d494c1] StructIO v0.3.0
  [c3572dad] Sundials v4.9.2
  [ab02a1b2] TableOperations v1.2.0
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.6.1
  [5d786b92] TerminalLoggers v0.1.5
  [8290d209] ThreadingUtilities v0.5.0
  [a759f4b9] TimerOutputs v0.5.15
  [9f7883ad] Tracker v0.2.20
  [28d57a85] Transducers v0.4.73
  [592b5752] Trapz v2.0.3
  [a2a6695c] TreeViews v0.3.0
  [d5829a12] TriangularSolve v0.1.11
  [fce5fe82] Turing v0.20.4
  [5c2747f8] URIs v1.3.0
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
  [41fe7b60] Unzip v0.1.2
  [3d5dd08c] VectorizationBase v0.21.25
  [19fa3120] VertexSafeGraphs v0.2.0
  [cc8bc4a8] Widgets v0.6.5
  [efce3f68] WoodburyMatrices v0.5.5
  [e88e6eb3] Zygote v0.6.35
  [700de1a5] ZygoteRules v0.2.2
  [68821587] Arpack_jll v3.5.0+3
  [6e34b625] Bzip2_jll v1.0.8+0
  [83423d85] Cairo_jll v1.16.1+1
  [5ae413db] EarCut_jll v2.2.3+0
  [7cc45869] Enzyme_jll v0.0.25+0
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
  [e33a78d0] Hwloc_jll v2.7.0+0
  [1d5cc7b8] IntelOpenMP_jll v2018.0.3+2
  [aacddb02] JpegTurbo_jll v2.1.2+0
  [c1c5ebd0] LAME_jll v3.100.1+0
  [88015f11] LERC_jll v3.0.0+1
  [dad2f222] LLVMExtra_jll v0.0.14+2
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
  [458c3c95] OpenSSL_jll v1.1.13+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.2+0
  [2f80f16e] PCRE_jll v8.44.0+0
  [30392449] Pixman_jll v0.40.1+0
  [ea2cea3b] Qt5Base_jll v5.15.3+0
  [f50d1b31] Rmath_jll v0.3.0+0
  [fb77eaff] Sundials_jll v5.2.0+1
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
  [8e850b90] libblastrampoline_jll v3.1.0+2
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
  [bea87d4a] SuiteSparse_jll
  [83775a58] Zlib_jll
  [8e850ede] nghttp2_jll
  [3f19e933] p7zip_jll
```

