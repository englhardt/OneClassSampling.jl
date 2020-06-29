"""
Zhu, F., Ye, N., Yu, W., Xu, S. and Li, G., 2014.
Boundary detection and sample reduction for one-class support vector machines.
Neurocomputing, 123, pp.166-173.

Neighbors’ Distribution Properties and Sample Reduction algorithm (NDPSR)
Parameter guideline from the paper, given a data set of size N:
k = 20
none for eps (on benchmark data eps ∈ [12-20])
"""
mutable struct NDPSR <: Sampler
    k::Int
    eps::Real
    function NDPSR(k::Int=20, eps::Real=15)
        k > 0 || throw(ArgumentError("Invalid number of neighbors to consider $(k)."))
        eps >= 0 || throw(ArgumentError("Invalid threshold $(eps)."))
        return new(k, eps)
    end
end

function sample(sampler::NDPSR, data::DataSet, labels::Vector{Symbol})::BitArray{1}
    sampler.k = min(size(data, 2) - 1, sampler.k)
    tree = KDTree(data)
    sample_mask = falses(size(data, 2))

    for i in 1:size(data, 2)
        idx, dist = knn(tree, data[:, i], sampler.k + 1, true)
        x_k = mean(data, dims=2)[:, 1]
        d_ik = normalize(data[:, i] .- x_k)
        d_ij = mapslices(normalize, data[:, i] .- data[:, idx[2:end]], dims=1)
        c_sum0 = sum(dot(d_ik, d_ij[:, j]) for j in 1:sampler.k)
        if c_sum0 >= sampler.eps
            sample_mask[i] = true
        end
    end
    return sample_mask
end
