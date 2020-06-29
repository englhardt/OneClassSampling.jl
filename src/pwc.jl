abstract type ThresholdStrategy end

struct FixedThresholdStrategy <: ThresholdStrategy
    threshold::Real
end

function calculate_threshold(threshold_strat::FixedThresholdStrategy, kde_cache::KDECache, labels::Vector{Symbol})
    return threshold_strat.threshold
end

struct NumLabelTresholdStrategy <: ThresholdStrategy
    eps::Real
    function NumLabelTresholdStrategy(eps::Real)
        eps >= 0 || throw(ArgumentError("Negative eps: $(eps)."))
        eps <= 1 || throw(ArgumentError("Eps bigger than 1: $(eps)."))
        new(eps)
    end
end

function calculate_threshold(threshold_strat::NumLabelTresholdStrategy, kde_cache::KDECache, labels::Vector{Symbol})
    return threshold_strat.eps * length(labels)
end

struct MaxDensityThresholdStrategy <: ThresholdStrategy
    eps::Real
    function MaxDensityThresholdStrategy(eps::Real)
        eps >= 0 || throw(ArgumentError("Negative eps: $(eps)."))
        eps <= 1 || throw(ArgumentError("Eps bigger than 1: $(eps)."))
        new(eps)
    end
end

function calculate_threshold(threshold_strat::MaxDensityThresholdStrategy, kde_cache::KDECache, labels::Vector{Symbol})
    return threshold_strat.eps * maximum(kde(kde_cache))
end

struct OutlierPercentageThresholdStrategy <: ThresholdStrategy
    eps::Union{Real, Nothing}
    function OutlierPercentageThresholdStrategy(eps::Union{Real, Nothing}=nothing)
        if eps !== nothing
            eps >= 0 || throw(ArgumentError("Negative eps: $(eps)."))
            eps <= 1 || throw(ArgumentError("Eps bigger than 1: $(eps)."))
        end
        new(eps)
    end
end

function calculate_threshold(threshold_strat::OutlierPercentageThresholdStrategy, kde_cache::KDECache, labels::Vector{Symbol})
    if threshold_strat.eps === nothing
        eps = count(x -> x .== :outlier, labels) / length(labels)
    else
        eps = threshold_strat.eps
    end
    d_x = sort(kde(kde_cache))
    n = length(labels)
    idx = max(1, min(n, floor(Int, eps * n)))
    return (d_x[idx] + d_x[min(n, idx+1)]) / 2
end

struct GroundTruthThresholdStrategy <: ThresholdStrategy end

function calculate_threshold(threshold_strat::GroundTruthThresholdStrategy, kde_cache::KDECache, labels::Vector{Symbol})
    eps = count(x -> x .== :outlier, labels) / length(labels)
    d_x = sort(kde(kde_cache))
    n = length(labels)
    idx = max(1, min(n, floor(Int, eps * n)))
    return (d_x[idx] + d_x[min(n, idx+1)]) / 2
end


function calculate_threshold(threshold_strat::ThresholdStrategy, data::DataSet, labels::Vector{Symbol},
                             gamma::Union{Real, Symbol})
    return calculate_threshold(threshold_strat, KDECache(data, gamma), labels)
end

function split_masks(threshold_strat::GroundTruthThresholdStrategy, kde_cache::KDECache, labels::Vector{Symbol})
    threshold = calculate_threshold(threshold_strat, kde_cache, labels)
    inlier_mask = labels .== :inlier
    outlier_mask = .!inlier_mask
    return threshold, inlier_mask, outlier_mask
end

function split_masks(threshold_strat::Nothing, kde_cache::KDECache, labels::Vector{Symbol})
    return 0.0, trues(length(labels)), falses(length(labels))
end

function split_masks(threshold_strat::ThresholdStrategy, kde_cache::KDECache, labels::Vector{Symbol})
    threshold = calculate_threshold(threshold_strat, kde_cache, labels)
    d_x = kde(kde_cache)
    inlier_mask = d_x .>= threshold
    outlier_mask = .!inlier_mask
    return threshold, inlier_mask, outlier_mask
end

function split_masks(threshold_strat::ThresholdStrategy, data::DataSet, labels::Vector{Symbol}, gamma::Union{Nothing, Real, Symbol}=nothing)
    return split_data(threshold_strat, KDECache(data, gamma), labels)
end

mutable struct PWC
    kde_cache::KDECache
    threshold_strat::ThresholdStrategy
    threshold::Union{Real, Nothing}
    function PWC(kde_cache::KDECache, threshold_strat::ThresholdStrategy=OutlierPercentageThresholdStrategy(),
                 threshold::Union{Real, Nothing}=nothing)
        return new(kde_cache, threshold_strat, threshold)
    end
end

function PWC(data, gamma::Union{Real, Symbol}, threshold_strat::ThresholdStrategy=OutlierPercentageThresholdStrategy())
    return PWC(KDECache(data, gamma), threshold_strat)
end

function calculate_threshold!(pwc::PWC, labels::Vector{Symbol})
    pwc.threshold = calculate_threshold(pwc.threshold_strat, pwc.kde_cache, labels)
    return nothing
end

function predict(pwc::PWC, data::DataSet)
    if pwc.threshold === nothing
        throw(ArgumentError("PWC threshold not initialized."))
    end
    d_x = kde(pwc.kde_cache, data)
    pred = fill(:outlier, size(data, 2))
    pred[d_x .> pwc.threshold] .= :inlier
    return pred
end
