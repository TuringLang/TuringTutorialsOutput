
@testset "PkgExtreme" begin
    include("arithmetic.jl")
    include("utils.jl")
end


# Yes:
@test value == 0

# No:
@test value == 0.0


# Check that m and s are plus or minus one from 1.5 and 2.2, respectively.
check_numerical(chain, [:m, :s], [1.5, 2.2]; atol=1.0)

# Checks the estimates for a default gdemo model using values 1.5 and 2.0.
check_gdemo(chain; atol=0.1)

# Checks the estimates for a default MoG model.
check_MoGtest_default(chain; atol=0.1)

