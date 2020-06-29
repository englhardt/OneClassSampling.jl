using OneClassSampling

import JuMP
import Ipopt
import MLKernels
import SVDD

using Random
using Test

Random.seed!(0)

TEST_SOLVER =  JuMP.with_optimizer(Ipopt.Optimizer, print_level=0)

@testset "OneClassSampling" begin
    include("sampler_test.jl")
    include("density_test.jl")
end
