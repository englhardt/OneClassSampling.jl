"""
Sun, W., Qu, J., Chen, Y., Di, Y. and Gao, F., 2016.
Heuristic sample reduction method for support vector data description.
Turkish Journal of Electrical Engineering & Computer Sciences, 24(1), pp.298-312.

Heuristic Sampling Reduction (HSR)
Parameter:
k
eps: 0.1 * num_dimensions
"""
mutable struct HSR <: Sampler
    k::Int
    eps::Real
    function HSR(k::Int, eps::Real=0.1)
        k > 0 || throw(ArgumentError("Invalid number of clusters $(k)."))
        eps >= 0 || throw(ArgumentError("Negative threshold parameter $(eps)."))
        return new(k, eps)
    end
end

function sample(sampler::HSR, data::DataSet, labels::Vector{Symbol})::BitArray{1}
    sampler.k = min(size(data, 2), sampler.k)
    threshold = sampler.eps * size(data, 1)
    clustering = kmeans(data, sampler.k)
    dist_center_sorted_idx = sortperm(clustering.costs)
    tree = KDTree(data)
    sample_mask = trues(size(data, 2))
    processed = falses(size(data, 2))
    for i in dist_center_sorted_idx
        if !processed[i]
            # inrange includes i itself
            # -> query sorted result and drop first element
            # Special case: another observation is equal to i
            # -> one of the two remains in sample_mask
            to_remove = inrange(tree, data[:, i], threshold, true)[2:end]
            sample_mask[to_remove] .= false
            processed[to_remove] .= true
            processed[i] = true
        end
    end
    @assert all(processed)
    return sample_mask
end
