"""
Xiao, Y., Liu, B., Hao, Z. and Cao, L., 2014.
A K-Farthest-Neighbor-based approach for support vector data description.
Applied intelligence, 41(1), pp.196-211.

K-Farthest-Neighbor-based Concept Boundary Detection (KFN-CBD)
Parameter guideline from the paper:
k = 100
eps (retaining 80% of the data)

Implementation Note:
1) We only implement the version with distance computation in the data space with the euclidean distance.
2) We do not use the proposed extension of the M-tree but naive farthest neighbor search with a KDTree.
"""
struct KFNCBD <: Sampler
    k::Int
    eps::Real
    function KFNCBD(k::Int=100, eps::Real=0.8)
        k > 0 || throw(ArgumentError("Invalid number of neighbors to consider $(k)."))
        (eps >= 0 && eps <= 1) || throw(ArgumentError("Invalid threshold $(eps)."))
        return new(k, eps)
    end
end

function sample(sampler::KFNCBD, data::DataSet, labels::Vector{Symbol})::BitArray{1}
    tree = KDTree(data)
    sample_mask = falses(size(data, 2))

    f_i = zeros(length(labels))
    for i in 1:size(data, 2)
        _, dist = knn(tree, data[:, i], length(labels), true)
        f_i[i] = 1 - 1 / sampler.k * sum(exp.(-dist[max(1, end-sampler.k+1):end]))
    end
    sample_mask = falses(size(data, 2))
    num_samples = floor(Int, length(labels) * sampler.eps)
    sample_mask[sortperm(f_i, rev=true)[1:num_samples]] .= true
    return sample_mask
end
