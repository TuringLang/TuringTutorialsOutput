
using Turing

# Define a simple Normal model with unknown mean and variance.
@model function gdemo(x, y)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    return y ~ Normal(m, sqrt(s²))
end

# Sample using Gibbs and varying autodiff backends.
c = sample(
    gdemo(1.5, 2),
    Gibbs(HMC{Turing.ForwardDiffAD{1}}(0.1, 5, :m), HMC{Turing.TrackerAD}(0.1, 5, :s²)),
    1000,
)

