"""
Chaudhuri, A., Kakde, D., Jahja, M., Xiao, W., Kong, S., Jiang, H. and Percdriy, S., 2018, January.
Sampling method for fast training of support vector data description.
In 2018 Annual Reliability and Maintainability Symposium (RAMS) (pp. 1-7). IEEE.

Python implementation available here: https://github.com/samplesvdd/sample_svdd/blob/master/sample_svdd.py

We dub the method random subsample SVDD (RSSVDD)

Parameter guideline from paper and code:
subsample_size = 10 (benchmark data ∈ [3, 20])
max_iter = 1000
convergence_stop_tol = 10^-6
convergence_num_iter = 30
"""
mutable struct RSSVDD <: Sampler
    subsample_size::Int
    max_iter::Int
    convergence_stop_tol::Real
    convergence_num_iter::Int
    init_strategy::SVDD.InitializationStrategy
    solver::OptimizerFactory
    function RSSVDD(subsample_size::Int, max_iter::Int, convergence_stop_tol::Real, convergence_num_iter::Int,
                    init_strategy::SVDD.InitializationStrategy, solver::OptimizerFactory)
        subsample_size > 0 || throw(ArgumentError("Invalid size of sub sample $(subsample_size)."))
        max_iter > 0 || throw(ArgumentError("Invalid number of iterations $(max_iter)."))
        convergence_stop_tol > 0 || throw(ArgumentError("Invalid connvergence tolerance $(convergence_stop_tol)."))
        convergence_num_iter > 0 || throw(ArgumentError("Invalid number of iterations for convergence $(convergence_num_iter)."))
        return new(subsample_size, max_iter, convergence_stop_tol, convergence_num_iter, init_strategy, solver)
    end
end

function RSSVDD(subsample_size::Int, max_iter::Int,
                init_strategy::SVDD.InitializationStrategy, solver::OptimizerFactory,
                convergence_stop_tol::Real=10^-6, convergence_num_iter::Int=30)
    return RSSVDD(subsample_size, max_iter, convergence_stop_tol, convergence_num_iter, init_strategy, solver)
end

function sample(sampler::RSSVDD, data::DataSet, labels::Vector{Symbol})::BitArray{1}
    n = size(data, 2)
    sampler.subsample_size = min(n, sampler.subsample_size)

    model = SVDD.VanillaSVDD(data)
    SVDD.initialize!(model, sampler.init_strategy)
    SVDD.set_adjust_K!(model, true)

    master_svs = []
    R = nothing
    num_stable_iter = 0
    for i in 1:sampler.max_iter
        try
            subsample_indices = randperm(n)[1:sampler.subsample_size]
            subsample_svs = train_and_find_support_vectors(model, data[:, subsample_indices], sampler.solver)
            master_svs = unique(vcat(master_svs, subsample_indices[subsample_svs]))
            new_master_svs = train_and_find_support_vectors(model, data[:, master_svs], sampler.solver)
            master_svs = master_svs[new_master_svs]
        catch e
            # Model fitting on subsample failed. Skipping subsample.
            continue
        end
        if R !== nothing && abs(model.R - R) <= sampler.convergence_stop_tol * abs(R)
            num_stable_iter += 1
        else
            num_stable_iter = 0
        end
        if num_stable_iter >= sampler.convergence_num_iter
            break
        end
        R = model.R
    end
    sample_mask = falses(n)
    sample_mask[master_svs] .= true
    return sample_mask
end
