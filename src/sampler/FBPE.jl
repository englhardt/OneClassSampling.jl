"""
Alam, S., Sonbhadra, S.K., Agarwal, S., Nagabhushan, P. and Tanveer, M., 2020.
Sample reduction using Farthest Boundary Point Estimation (FBPE) for Support Vector Data Description (SVDD).
Pattern Recognition Letters.

Farthest Boundary Point Estimation (FBPE)
Parameter:
n: number of directions to check (no stepsize recommendation for the full sweep from 0 to 2π )

To compute the angle we mean center the data (so mean is 0 vector).
Then we select a random data point as reference point and compute the angle to all other points.
All these angles are then binned into n bins of equal size.
With cos-sim ∈ [-1, 1], the value range covered by each bin is 2 / n large.
"""
struct FBPE <: Sampler
    n::Int
    function FBPE(n::Int)
        n > 0 || throw(ArgumentError("Non-positive number of directions to check $(n)."))
        return new(n)
    end
end

function sample(sampler::FBPE, data::DataSet, labels::Vector{Symbol})::BitArray{1}
    rejected = falses(size(data, 2))
    F_x = Int[]
    while !all(rejected)
        remaining_indices = findall(.!rejected)
        if length(remaining_indices) == 1
            push!(F_x, first(remaining_indices))
            break
        end
        # Normalize data so mean is the center and we can perform a sweep in n directions
        x_mean = mean(data[:, remaining_indices], dims=2)
        data_centered = data[:, remaining_indices] .- x_mean
        x_ref = data_centered[:, 1]
        # compute cosine similarity
        similarity = [cosine_similarity(x_ref, data_centered[:, i]) for i in 1:size(data_centered, 2)]
        # Iterate over all directions
        S_x = []
        bin_thresholds = collect(range(-1 - 10^-5, 1, length=sampler.n + 1))
        for t in 2:length(bin_thresholds)
            bin_indices = findall((similarity .> bin_thresholds[t-1]) .& (similarity .<= bin_thresholds[t]))
            if !isempty(bin_indices)
                max_dist, max_idx = findmax([norm(data_centered[:, i]) for i in bin_indices])
                new_reject_indices = [i for i in bin_indices if i != bin_indices[max_idx]]
                # only keep farthest point, reject others
                rejected[remaining_indices[new_reject_indices]] .= true
                push!(S_x, (max_dist, remaining_indices[bin_indices[max_idx]]))
            end
        end
        # select farthest point of all farthest points per direction
        sort!(S_x, rev=true)
        farthest_idx = first(S_x)[2]
        rejected[farthest_idx] = true
        push!(F_x, farthest_idx)
    end
    sample_mask = falses(size(data, 2))
    sample_mask[F_x] .= true
    return sample_mask
end
