mutable struct RAPID <: Sampler
    threshold_strat::ThresholdStrategy
    threshold::Union{Real, Nothing}
    kde_cache::Union{KDECache, Nothing}
    gamma::Union{Symbol, Real}
    function RAPID(threshold_strat::ThresholdStrategy, kde_cache::Union{KDECache, Nothing}, gamma::Union{Symbol, Real})
        return new(threshold_strat, nothing, kde_cache, gamma)
    end
end

function RAPID(threshold_strat::ThresholdStrategy, kde_cache::KDECache)
    return RAPID(threshold_strat, kde_cache, :precomputed)
end

function RAPID(threshold_strat::ThresholdStrategy, gamma::Union{Symbol, Real})
    return RAPID(threshold_strat, nothing, gamma)
end

function sample(sampler::RAPID, data::DataSet, labels::Vector{Symbol})::BitArray{1}
    if sampler.kde_cache === nothing || sampler.kde_cache.data != data
        sampler.kde_cache = KDECache(data, sampler.gamma)
    end
    sampler.threshold, inlier_mask, outlier_mask = split_masks(sampler.threshold_strat,
                                                               sampler.kde_cache,
                                                               labels)

    remove_order = zeros(Int, size(data, 2))
    sample_mask = copy(inlier_mask)
    d_x = kde(sampler.kde_cache, sample_mask, trues(size(data, 2)))
    i = 1
    num_iter = count(sample_mask) - 2
    while i < num_iter
        idx = findall(sample_mask)[findmax(d_x[sample_mask])[2]]
        sample_mask[idx] = false
        d_x -= sampler.kde_cache.K[:, idx]
        θ = minimum(d_x[sample_mask])
        if !all(d_x[inlier_mask] .>= θ)
            break
        end
        remove_order[i] = idx
        i += 1
    end
    sample_mask = inlier_mask
    sample_mask[remove_order[1:i-1]] .= false
    sampler.threshold = any(sample_mask) ? minimum(kde(sampler.kde_cache, sample_mask, sample_mask)) : 0
    return sample_mask
end
