"""
Qu, H., Zhao, J., Zhao, J. and Jiang, D., 2019, June.
Towards support vector data description based on heuristic sample condensed rule.
In 2019 Chinese Control And Decision Conference (CCDC) (pp. 4647-4653). IEEE.

Heuristic Sample Condensed (HSC)
Parameter:
k for kNN search: no recommendation
n_grid: number of parameters to test for each threshold during grid search

We use a PWC to split the data set into inlier/outlier to
allow tuning the disT and denT parameters from the paper.
"""
mutable struct HSC <: Sampler
    k::Int
    n_grid::Int
    threshold_strat::ThresholdStrategy
    threshold::Union{Real, Nothing}
    kde_cache::Union{KDECache, Nothing}
    gamma::Union{Symbol, Real}
    init_strategy::SVDD.InitializationStrategy
    solver::OptimizerFactory
    function HSC(k::Int, n_grid::Int,
                 threshold_strat::ThresholdStrategy, kde_cache::Union{KDECache, Nothing}, gamma::Union{Symbol, Real},
                 init_strategy::SVDD.InitializationStrategy, solver::OptimizerFactory)
        k > 0 || throw(ArgumentError("Non-positive number of nearest neighbors to consider $(k)."))
        n_grid > 0 || throw(ArgumentError("Non-positive number of grid parameter to test per threshold $(n_grid)."))
        return new(k, n_grid, threshold_strat, nothing, kde_cache, gamma, init_strategy, solver)
    end
end

function HSC(k::Int, n_grid::Int,
             threshold_strat::ThresholdStrategy, kde_cache::KDECache,
             init_strategy::SVDD.InitializationStrategy, solver::OptimizerFactory)
    return HSC(k, n_grid, threshold_strat, kde_cache, :precomputed, init_strategy, solver)
end

function HSC(k::Int, n_grid::Int,
             threshold_strat::ThresholdStrategy, gamma::Union{Symbol, Real},
             init_strategy::SVDD.InitializationStrategy, solver::OptimizerFactory)
    return HSC(k, n_grid, threshold_strat, nothing, gamma, init_strategy, solver)
end

function calculate_hsc_values(sampler, data)
    n = size(data, 2)
    sampler.k = min(n - 1, sampler.k)
    tree = KDTree(data)
    p_i = zeros(n)
    indices, distances = knn(tree, data, sampler.k + 1, true)
    p_i = [exp(-mean(distances[i][2:end])) for i in 1:n]
    d_ij = Distances.pairwise(Distances.Euclidean(), data, dims=2)
    δ_i = zeros(n)
    for i in 1:n
        dist_to_high_dens = [d_ij[i, j] for j in 1:n if p_i[j] > p_i[i]]
        δ_i[i] = isempty(dist_to_high_dens) ? maximum(d_ij[i, :]) : minimum(dist_to_high_dens)
    end
    return δ_i, p_i
end

function hsc_mask(δ_i, p_i, disT, denT)
    mask = falses(length(p_i))
    mask[(δ_i .<= disT) .& (p_i .<= denT)] .= true
    return mask
end

function sample(sampler::HSC, data::DataSet, labels::Vector{Symbol})::BitArray{1}
    δ_i, p_i = calculate_hsc_values(sampler, data)

    # Split data set to allow disT and denT tuning
    if sampler.kde_cache === nothing || sampler.kde_cache.data != data
        sampler.kde_cache = KDECache(data, sampler.gamma)
    end
    pwc = PWC(sampler.kde_cache, sampler.threshold_strat)
    calculate_threshold!(pwc, labels)
    pwc_ground_truth = predict(pwc, data)

    # Tune disT and denT
    best_s = -Inf
    best_mask = falses(size(data, 2))
    for disT in range(extrema(δ_i)..., length=sampler.n_grid)
        for denT in range(extrema(p_i)..., length=sampler.n_grid)
            sample_mask = hsc_mask(δ_i, p_i, disT, denT)
            if !any(sample_mask)
                continue
            end
            model = SVDD.VanillaSVDD(data)
            SVDD.initialize!(model, sampler.init_strategy)
            SVDD.set_adjust_K!(model, true)
            SVDD.set_data!(model, data[:, sample_mask])
            try
                SVDD.fit!(model, sampler.solver)
            catch e
                debug(LOGGER, "[HSC] Failed fitting model on data set of size $(size(model.data)) due to $e.")
                continue
            end
            pred = SVDD.predict(model, data)
            s = mcc(pred, pwc_ground_truth)
            if s > best_s
                best_s = s
                best_mask = copy(sample_mask)
            end
        end
    end
    if best_s == -Inf
        throw(SamplingException("[HSC] Failed fitting any model."))
    end
    return best_mask
end
