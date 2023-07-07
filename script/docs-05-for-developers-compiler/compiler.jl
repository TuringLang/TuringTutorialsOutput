
(model::Model)([rng, varinfo, sampler, context])


@model function gauss(
    x=missing, y=1.0, ::Type{TV}=Vector{Float64}
) where {TV<:AbstractVector}
    if x === missing
        x = TV(undef, 3)
    end
    p = TV(undef, 2)
    p[1] ~ InverseGamma(2, 3)
    p[2] ~ Normal(0, 1.0)
    @. x[1:2] ~ Normal(p[2], sqrt(p[1]))
    x[3] ~ Normal()
    return y ~ Normal(p[2], sqrt(p[1]))
end


@model function gauss(
    ::Type{TV}=Vector{Float64}; x=missing, y=1.0
) where {TV<:AbstractVector}
    return ...
end


if x === missing
    x = ...
end


@model function gauss(x, y=1.0, ::Type{TV}=Vector{Float64}) where {TV<:AbstractVector}
    p = TV(undef, 2)
    return ...
end

function gauss(::Missing, y=1.0, ::Type{TV}=Vector{Float64}) where {TV<:AbstractVector}
    return gauss(TV(undef, 3), y, TV)
end


#= REPL[25]:6 =#
begin
    var"##tmpright#323" = InverseGamma(2, 3)
    var"##tmpright#323" isa Union{Distribution,AbstractVector{<:Distribution}} || throw(
        ArgumentError(
            "Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions.",
        ),
    )
    var"##vn#325" = (DynamicPPL.VarName)(:p, ((1,),))
    var"##inds#326" = ((1,),)
    p[1] = (DynamicPPL.tilde_assume)(
        _rng,
        _context,
        _sampler,
        var"##tmpright#323",
        var"##vn#325",
        var"##inds#326",
        _varinfo,
    )
end


#= REPL[25]:8 =#
begin
    var"##tmpright#331" = Normal.(p[2], sqrt.(p[1]))
    var"##tmpright#331" isa Union{Distribution,AbstractVector{<:Distribution}} || throw(
        ArgumentError(
            "Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions.",
        ),
    )
    var"##vn#333" = (DynamicPPL.VarName)(:x, ((1:2,),))
    var"##inds#334" = ((1:2,),)
    var"##isassumption#335" = begin
        let var"##vn#336" = (DynamicPPL.VarName)(:x, ((1:2,),))
            if !((DynamicPPL.inargnames)(var"##vn#336", _model)) ||
                (DynamicPPL.inmissings)(var"##vn#336", _model)
                true
            else
                x[1:2] === missing
            end
        end
    end
    if var"##isassumption#335"
        x[1:2] .= (DynamicPPL.dot_tilde_assume)(
            _rng,
            _context,
            _sampler,
            var"##tmpright#331",
            x[1:2],
            var"##vn#333",
            var"##inds#334",
            _varinfo,
        )
    else
        (DynamicPPL.dot_tilde_observe)(
            _context,
            _sampler,
            var"##tmpright#331",
            x[1:2],
            var"##vn#333",
            var"##inds#334",
            _varinfo,
        )
    end
end


struct VarInfo{Tmeta, Tlogp} <: AbstractVarInfo
    metadata::Tmeta
    logp::Base.RefValue{Tlogp}
    num_produce::Base.RefValue{Int}
end

