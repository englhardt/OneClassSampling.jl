"""
Li, Yuhua, 2011.
Selecting training points for one-class support vector machines.
Pattern recognition letters 32.11: 1517-1522.

We dub the method boundary point selector (BPS).

Parameter guideline from the paper, given a data set of size N:
k = floor(10 ln N)
eps ∈ [0, 0.2]
"""
mutable struct BPS <: Sampler
    k::Union{Int, Nothing}
    eps::Real
    function BPS(k::Union{Int, Nothing}=nothing, eps::Real=0.2)
        isa(k, Int) && (k > 0 || throw(ArgumentError("Invalid number of neighbors to consider $(k).")))
        eps >= 0 || throw(ArgumentError("Invalid threshold $(eps)."))
        return new(k, eps)
    end
end

function sample(sampler::BPS, data::DataSet, labels::Vector{Symbol})::BitArray{1}
    if sampler.k === nothing
        sampler.k = round(Int, floor(10 * log(size(data, 2))))
    end
    sampler.k = min(size(data, 2) - 1, sampler.k)
    tree = KDTree(data)
    sample_mask = falses(size(data, 2))

    for i in 1:size(data, 2)
        idx, dist = knn(tree, data[:, i], sampler.k + 1, true)
        v_ij = mapslices(normalize, data[:, i] .- data[:, idx[2:end]], dims=1)
        n_i = sum(v_ij, dims=2)
        θ_ij = sum(v_ij .* n_i, dims=1)
        l_i = 1 / sampler.k * sum(θ_ij .>= 0)
        if l_i >= 1 - sampler.eps
            sample_mask[i] = true
        end
    end
    return sample_mask
end
