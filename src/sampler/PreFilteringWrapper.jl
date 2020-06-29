mutable struct PreFilteringWrapper <: Sampler
    sampler::Sampler
    threshold_strat::Union{ThresholdStrategy, Nothing}
    threshold::Union{Real, Nothing}
    kde_cache::Union{KDECache, Nothing}
    gamma::Union{Symbol, Real}
    function PreFilteringWrapper(sampler::Sampler, kde_cache::Union{KDECache, Nothing}, gamma::Union{Symbol, Real}, threshold_strat::Union{ThresholdStrategy, Nothing}=nothing)
        return new(sampler, threshold_strat, nothing, kde_cache, gamma)
    end
end

function PreFilteringWrapper(sampler::Sampler, kde_cache::KDECache, threshold_strat::Union{ThresholdStrategy, Nothing}=nothing)
    return PreFilteringWrapper(sampler, kde_cache, :precomputed, threshold_strat)
end

function PreFilteringWrapper(sampler::Sampler, gamma::Union{Symbol, Real}, threshold_strat::Union{ThresholdStrategy, Nothing}=nothing)
    return PreFilteringWrapper(sampler, nothing, gamma, threshold_strat)
end

function sample(sampler::PreFilteringWrapper, data::DataSet, labels::Vector{Symbol})::BitArray{1}
    if sampler.threshold_strat === nothing
        return sample(sampler.sampler, data, labels)
    end
    if sampler.kde_cache === nothing || sampler.kde_cache.data != data
        sampler.kde_cache = KDECache(data, sampler.gamma)
    end
    debug(LOGGER, "Prepruning data set with density threshold $(sampler.threshold_strat).")
    sampler.threshold, inlier_mask, outlier_mask = split_masks(sampler.threshold_strat,
                                                               sampler.kde_cache,
                                                               labels)
    sample_mask = inlier_mask
    sub_mask = sample(sampler.sampler, data[:, sample_mask], labels[sample_mask])
    sample_mask[sample_mask] = sub_mask
    return sample_mask
end
