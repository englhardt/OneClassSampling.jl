mutable struct HDS <: Sampler
    target_ratio::Real
    threshold_strat::Union{ThresholdStrategy, Nothing}
    threshold::Union{Real, Nothing}
    kde_cache::Union{KDECache, Nothing}
    gamma::Union{Symbol, Real}
    function HDS(target_ratio::Real, kde_cache::Union{KDECache, Nothing}, gamma::Union{Symbol, Real}, threshold_strat::Union{ThresholdStrategy, Nothing}=nothing)
        0 <= target_ratio <= 1 || throw(ArgumentError("Invalid sampling ratio $(target_ratio)."))
        return new(target_ratio, threshold_strat, nothing, kde_cache, gamma)
    end
end

function HDS(target_ratio::Real, kde_cache::KDECache, threshold_strat::Union{ThresholdStrategy, Nothing}=nothing)
    return HDS(target_ratio, kde_cache, :precomputed, threshold_strat)
end

function HDS(target_ratio::Real, gamma::Union{Symbol, Real}, threshold_strat::Union{ThresholdStrategy, Nothing}=nothing)
    return HDS(target_ratio, nothing, gamma, threshold_strat)
end

function sample(sampler::HDS, data::DataSet, labels::Vector{Symbol})::BitArray{1}
    if sampler.kde_cache === nothing || sampler.kde_cache.data != data
        sampler.kde_cache = KDECache(data, sampler.gamma)
    end
    sampler.threshold, inlier_mask, outlier_mask = split_masks(sampler.threshold_strat,
                                                               sampler.kde_cache,
                                                               labels)
    sample_mask = inlier_mask
    target_size = round(Int, sampler.target_ratio * size(data, 2))
    d_x = kde(sampler.kde_cache, sample_mask, trues(size(data, 2)))
    while count(sample_mask) > target_size
        idx = findall(sample_mask)[findmax(d_x[sample_mask])[2]]
        sample_mask[idx] = false
        d_x -= sampler.kde_cache.K[:, idx]
    end
    sampler.threshold = any(sample_mask) ? minimum(kde(sampler.kde_cache, sample_mask, sample_mask)) : 0
    return sample_mask
end
