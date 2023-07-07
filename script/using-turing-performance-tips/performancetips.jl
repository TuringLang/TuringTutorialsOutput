
using Turing
@model function gmodel(x)
    m ~ Normal()
    for i in 1:length(x)
        x[i] ~ Normal(m, 0.2)
    end
end


using FillArrays

@model function gmodel(x)
    m ~ Normal()
    return x ~ MvNormal(Fill(m, length(x)), 0.04 * I)
end


@model function tmodel(x, y)
    p, n = size(x)
    params = Vector{Real}(undef, n)
    for i in 1:n
        params[i] ~ truncated(Normal(), 0, Inf)
    end

    a = x * params
    return y ~ MvNormal(a, I)
end


@model function tmodel(x, y, ::Type{T}=Float64) where {T}
    p, n = size(x)
    params = Vector{T}(undef, n)
    for i in 1:n
        params[i] ~ truncated(Normal(), 0, Inf)
    end

    a = x * params
    return y ~ MvNormal(a, I)
end


@model function tmodel(x, y)
    params ~ filldist(truncated(Normal(), 0, Inf), size(x, 2))
    a = x * params
    return y ~ MvNormal(a, I)
end


@model function tmodel(x)
    p = Vector{Real}(undef, 1)
    p[1] ~ Normal()
    p = p .+ 1
    return x ~ Normal(p[1])
end


using Random

model = tmodel(1.0)

@code_warntype model.f(
    model,
    Turing.VarInfo(model),
    Turing.SamplingContext(
        Random.GLOBAL_RNG, Turing.SampleFromPrior(), Turing.DefaultContext()
    ),
    model.args...,
)


@model function demo(x)
    a ~ Gamma()
    b ~ Normal()
    c = function1(a)
    d = function2(b)
    return x .~ Normal(c, d)
end
alg = Gibbs(MH(:a), MH(:b))
sample(demo(zeros(10)), alg, 1000)


using Memoization

@memoize memoized_function1(args...) = function1(args...)
@memoize memoized_function2(args...) = function2(args...)


@model function demo(x)
    a ~ Gamma()
    b ~ Normal()
    c = memoized_function1(a)
    d = memoized_function2(b)
    return x .~ Normal(c, d)
end



