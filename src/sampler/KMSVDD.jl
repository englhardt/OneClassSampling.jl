"""
Kim, P.J., Chang, H.J., Song, D.S. and Choi, J.Y., 2007, June.
Fast support vector data description using k-means clustering.
In International Symposium on Neural Networks (pp. 506-514). Springer, Berlin, Heidelberg.

We dub the method K-Means subsamples for SVDD (KMSVDD)
Parameters:
on benchmark data k ∈ [5, 10, 20, 30, 40, 50]
init strategy and solver for SVDD solution
"""
struct KMSVDD <: Sampler
    k::Int
    init_strategy::SVDD.InitializationStrategy
    solver::OptimizerFactory
    function KMSVDD(k::Int, init_strategy::SVDD.InitializationStrategy, solver::OptimizerFactory)
        k > 0 || throw(ArgumentError("Invalid number of clusters $(k)."))
        return new(k, init_strategy, solver)
    end
end

function sample(sampler::KMSVDD, data::DataSet, labels::Vector{Symbol})::BitArray{1}
    clustering = kmeans(data, sampler.k)
    model = SVDD.VanillaSVDD(data)
    SVDD.initialize!(model, sampler.init_strategy)
    SVDD.set_adjust_K!(model, true)

    sample_mask = falses(size(data, 2))
    for i in 1:sampler.k
        cluster_indices = findall(clustering.assignments .== i)
        svs = train_and_find_support_vectors(model, data[:, cluster_indices], sampler.solver)
        sample_mask[cluster_indices[svs]] .= true
    end
    sv_mask = findall(sample_mask)
    svs = train_and_find_support_vectors(model, data[:, sv_mask], sampler.solver)
    sample_mask = falses(size(data, 2))
    sample_mask[sv_mask[svs]] .= true
    return sample_mask
end
