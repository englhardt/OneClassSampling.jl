using Clustering
using Distances
using JuMP
using MLKernels
using NearestNeighbors
using SVDD

using LinearAlgebra
using Statistics
using Random

abstract type Sampler end

DataSet = Array{Float64, 2}

function sample(sampler::Sampler, data::DataSet, labels::Vector{Symbol})::BitArray{1} end

struct SamplingException <: Exception
    msg
end

Base.showerror(io::IO, e::SamplingException) = print(io, "Exception during sampling: $(e.msg)")

include("density_util.jl")
include("pwc.jl")

include("sampler_util.jl")
include("sampler/RandomSampler.jl")
include("sampler/HDS.jl")

include("sampler/RAPID.jl")

include("sampler/BPS.jl")
include("sampler/CSSVDD.jl")
include("sampler/DAEDS.jl")
include("sampler/DBSRSVDD.jl")
include("sampler/FBPE.jl")
include("sampler/HSC.jl")
include("sampler/HSR.jl")
include("sampler/IESRSVDD.jl")
include("sampler/KFNCBD.jl")
include("sampler/KMSVDD.jl")
include("sampler/NDPSR.jl")
include("sampler/RSSVDD.jl")

include("sampler/PreFilteringWrapper.jl")
