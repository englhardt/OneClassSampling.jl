"""
Chu, C.S., Tsang, I.W. and Kwok, J.T., 2004, July.
Scaling up support vector data description by using core-sets.
In 2004 IEEE International Joint Conference on Neural Networks (IEEE Cat. No. 04CH37541) (Vol. 1, pp. 425-430). IEEE.

We dub the method core-set SVDD (CSSVDD)

Parameter guideline from paper:
num_init_sample = 20
k = 10
delta = 0.01*eps
eps none (experiments eps ∈ [0.15, 0.2, 0.3, 0.4, 0.5])
init_strategy: any strategy for gamma, C=1

Implementation notes:
1) We do not use fast incremental SVDD updates but fit a model from scratch each
   iteration.
2) We fit the initial SVDD not only on the approximal center and farthest point
   from the initial sample but on all points of the initial sample to avoid
   solver errors.
"""
mutable struct CSSVDD <: Sampler
    num_init_sample::Int
    k::Int
    delta::Real
    eps::Real
    outlier_percentage::Real
    init_strategy::SVDD.InitializationStrategy
    solver::OptimizerFactory
    function CSSVDD(num_init_sample::Int, k::Int, delta::Real, eps::Real, outlier_percentage::Real,
                    init_strategy::SVDD.InitializationStrategy, solver::OptimizerFactory)
        num_init_sample > 0 || throw(ArgumentError("Non-positive size of initial sample $(num_init_sample)."))
        k > 0 || throw(ArgumentError("Non-positive value k to initialize R $(k)."))
        delta > 0 || throw(ArgumentError("Non-positive value for delta $(delta)."))
        eps > 0 || throw(ArgumentError("Non-positive value for eps $(eps)."))
        outlier_percentage >= 0 || outlier_percentage <= 1 || throw(ArgumentError("Outlier percentage not in [0, 1] $(outlier_percentage)."))
        return new(num_init_sample, k, delta, eps, outlier_percentage, init_strategy, solver)
    end
end

function CSSVDD(eps::Real, outlier_percentage::Real, init_strategy::SVDD.InitializationStrategy, solver::OptimizerFactory,
                num_init_sample::Int=20, k::Int=10)
    return CSSVDD(num_init_sample, k, 0.01*eps, eps, outlier_percentage, init_strategy, solver)
end

function sample(sampler::CSSVDD, data::DataSet, labels::Vector{Symbol})::BitArray{1}
    n = size(data, 2)
    sampler.num_init_sample = min(n, sampler.num_init_sample)
    # Setup model
    model = SVDD.VanillaSVDD(data)
    SVDD.initialize!(model, sampler.init_strategy)
    SVDD.set_adjust_K!(model, true)
    K = copy(model.K_adjusted)
    # Step 1: Initialize R and c
    # Choose initial random sample and learn SVDD
    initial_sample = randperm(size(data, 2))[1:sampler.num_init_sample]
    SVDD.set_data!(model, data[:, initial_sample])
    try
        SVDD.fit!(model, sampler.solver)
    catch e
        throw(SamplingException("Failed fitting model on data set of size $(size(model.data)) due to $e."))
    end
    # Find closest point to center
    c = initial_sample[findmin(SVDD.predict(model, data[:, initial_sample]))[2]]
    # Set working set S (see implementation notes)
    S = copy(initial_sample)
    # Find furthest point from the random point (point 1) in the sample to init R
    # Distance is computed in the kernel space
    R_prev = minimum(K[:, initial_sample[1]]) / sampler.k
    while true
        # Step 2: Find set of points outside of (1+eps)-ball
        if length(S) == 1
            P = K[:, c] .> (1 + sampler.eps) * R_prev
        else
            model.R = (1 + sampler.eps) * model.R
            P = SVDD.predict(model, data) .> 0
            R_prev = model.R
        end
        # Step 3: check termination criteria
        if count(P) <= sampler.outlier_percentage * n
            break
        end
        # Step 4: Expand core-set with closest point to c in P that is not in S
        P_indices = [i for i in findall(P) if i ∉ S]
        if isempty(P_indices)
            break
        end
        p_close_out = P_indices[findmin(K[P_indices, c])[2]]
        push!(S, p_close_out)
        # Step 5: refit SVDD
        SVDD.set_data!(model, data[:, S])
        try
            SVDD.fit!(model, sampler.solver)
        catch e
            throw(SamplingException("Failed fitting model on data set of size $(size(model.data)) due to $e."))
        end
        # Step 6: enforce radius constraint
        if model.R < (1 + sampler.delta * sampler.eps) * R_prev
            model.R += sampler.delta * sampler.eps
        end
    end
    sample_mask = falses(n)
    sample_mask[S] .= true
    return sample_mask
end
